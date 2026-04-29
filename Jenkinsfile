pipeline {
  agent any

  options {
    timestamps()
    buildDiscarder(logRotator(numToKeepStr: '10'))
    skipDefaultCheckout()
  }

  environment {
    DOCKER_REPO = 'daviddwyer1987/qmd-cicadia'
    DOCKER_TAG = ''
  }

  stages {
    stage('Get Next Docker Tag') {
      steps {
        script {
          sh '''
            echo "Fetching tags from Docker Hub for ${DOCKER_REPO}..."
            curl -s "https://hub.docker.com/v2/repositories/${DOCKER_REPO}/tags?page_size=100" > tags.json
            cat tags.json
            LATEST=$(cat tags.json | grep -oE '"name":"[0-9]+\\.[0-9]+\\.[0-9]+"' | sed 's/"name":"//;s/"//' | sort -t. -k1,1n -k2,2n -k3,3n | tail -1)
            if [ -z "$LATEST" ]; then
              NEXT_TAG="0.0.1"
            else
              MAJOR=$(echo $LATEST | cut -d. -f1)
              MINOR=$(echo $LATEST | cut -d. -f2)
              PATCH=$(echo $LATEST | cut -d. -f3)
              PATCH=$((PATCH + 1))
              NEXT_TAG="${MAJOR}.${MINOR}.${PATCH}"
            fi
            echo "Latest tag: $LATEST"
            echo "Next tag: $NEXT_TAG"
            echo "$NEXT_TAG" > docker-tag.txt
          '''
          env.DOCKER_TAG = readFile('docker-tag.txt').trim()
          echo "Setting DOCKER_TAG to: ${env.DOCKER_TAG}"
        }
      }
    }

    stage('Checkout') {
      steps {
        checkout([
          $class: 'GitSCM',
          branches: [[name: '*/main']],
          userRemoteConfigs: [[url: 'https://github.com/cicadialabs/qmd.git']]
        ])
        sh '''
          git rev-parse --short HEAD > revision.txt
          echo "Cloned revision: $(cat revision.txt)"
          echo "Files in workspace:"
          ls -la
          echo "Docker tag for this build: ${DOCKER_TAG}"
        '''
      }
    }

    stage('Docker Build') {
      steps {
        script {
          sh '''
            echo "Building Docker image..."
            echo "Repository: ${DOCKER_REPO}"
            echo "Tag: ${DOCKER_TAG}"
            docker build -t ${DOCKER_REPO}:${DOCKER_TAG} -t ${DOCKER_REPO}:latest .
            echo "Docker image built successfully!"
            docker images | grep qmd-cicadia
          '''
        }
      }
    }

    stage('Docker Push') {
      steps {
        withCredentials([usernamePassword(credentialsId: 'docker-hub-credentials', usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {
          sh '''
            echo "Logging into Docker Hub..."
            echo "$DOCKER_PASS" | docker login -u "$DOCKER_USER" --password-stdin
            echo "Pushing Docker image..."
            docker push ${DOCKER_REPO}:${DOCKER_TAG}
            docker push ${DOCKER_REPO}:latest
            echo "Docker image pushed successfully!"
            echo "Image: ${DOCKER_REPO}:${DOCKER_TAG}"
            docker logout
          '''
        }
      }
    }
  }

  post {
    success {
      echo "Pipeline completed successfully! Docker image pushed as: ${env.DOCKER_TAG}"
    }
    failure {
      echo "Pipeline failed!"
    }
  }
}
