pipeline {
    agent any

    environment {
        DEPLOYMENT_SERVER = '192.168.20.235'
        DEPLOYMENT_DIRECTORY = '~/firepulse-detection-engine'
        IMAGE_TAG = 'latest'
    }

    stages {
        stage('Transfer Configuration') {
            steps {
                script {
                    withCredentials([
                        string(credentialsId: 'ssh-user', variable: 'SSH_USER'),
                        string(credentialsId: 'ssh-password', variable: 'SSH_PASSWORD'),
                        string(credentialsId: 'remote-api-base-url', variable: 'REMOTE_API_BASE_URL'),
                        string(credentialsId: 'remote-api-email', variable: 'REMOTE_API_EMAIL'),
                        string(credentialsId: 'remote-api-password', variable: 'REMOTE_API_PASSWORD'),
                        string(credentialsId: 'remote-images-serve-base-url', variable: 'REMOTE_IMAGES_SERVE_BASE_URL')
                    ]) {
                        // Create needed directories if not exist and transfer .env file
                        sh """
                            sshpass -p "${SSH_PASSWORD}" ssh -o StrictHostKeyChecking=no ${SSH_USER}@${DEPLOYMENT_SERVER} '
                                mkdir -p ${DEPLOYMENT_DIRECTORY}/data/{live,raw}
                                echo "REMOTE_API_BASE_URL=${REMOTE_API_BASE_URL}" > ${DEPLOYMENT_DIRECTORY}/.env
                                echo "REMOTE_API_EMAIL=${REMOTE_API_EMAIL}" >> ${DEPLOYMENT_DIRECTORY}/.env
                                echo "REMOTE_API_PASSWORD=${REMOTE_API_PASSWORD}" >> ${DEPLOYMENT_DIRECTORY}/.env
                                echo "REMOTE_IMAGES_SERVE_BASE_URL=${REMOTE_IMAGES_SERVE_BASE_URL}" >> ${DEPLOYMENT_DIRECTORY}/.env
                                chmod 600 ${DEPLOYMENT_DIRECTORY}/.env
                            '
                        """
                        // Transfer docker-compose.yaml
                        sh """
                            sshpass -p "${SSH_PASSWORD}" scp -o StrictHostKeyChecking=no docker-compose.yaml ${SSH_USER}@${DEPLOYMENT_SERVER}:${DEPLOYMENT_DIRECTORY}/
                        """
                    }
                }
            }
        }

        stage('Deploy Service') {
            steps {
                script {
                    withCredentials([
                        string(credentialsId: 'ssh-user', variable: 'SSH_USER'),
                        string(credentialsId: 'ssh-password', variable: 'SSH_PASSWORD'),
                        string(credentialsId: 'github-token', variable: 'GITHUB_TOKEN')
                    ]) {
                        try {
                            sh """
                                sshpass -p "${SSH_PASSWORD}" ssh -o StrictHostKeyChecking=no ${SSH_USER}@${DEPLOYMENT_SERVER} '
                                    set -e
                                    echo "${GITHUB_TOKEN}" | docker login ghcr.io -u "jenkins" --password-stdin
                                    cd ${DEPLOYMENT_DIRECTORY}
                                    export IMAGE_TAG=${IMAGE_TAG}
                                    echo "Pulling updated image for service..."
                                    docker compose pull web-service
                                    echo "Starting service..."
                                    docker compose up -d web-service --wait
                                    echo "Service deployed successfully!"
                                '
                            """
                        } catch (err) {
                            echo "Deployment process encountered an error: ${err}"
                            error(err.toString())
                        } finally {
                            // always attempt to clean up images and logout from registry
                            sh """
                                sshpass -p "${SSH_PASSWORD}" ssh -o StrictHostKeyChecking=no ${SSH_USER}@${DEPLOYMENT_SERVER} '
                                    docker image prune -f || true
                                    docker logout ghcr.io || true
                                '
                            """
                        }
                    }
                }
            }
        }
    }

    post {
        success {
            echo "✓ Deployment completed successfully!"
        }
        failure {
            echo "✗ Deployment failed!"
        }
        always {
            cleanWs()
        }
    }
}
