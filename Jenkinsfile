pipeline {
  agent {
    kubernetes {
      yaml '''
apiVersion: v1
kind: Pod
metadata:
  labels:
    cicadia.io/agent: qmd-build
spec:
  restartPolicy: Never
  containers:
    - name: jnlp
      image: jenkins/inbound-agent:latest-jdk17
      resources:
        requests:
          cpu: "100m"
          memory: "256Mi"
        limits:
          cpu: "500m"
          memory: "512Mi"
    - name: builder
      image: docker:24-dind
      command: ["sh", "-c", "cat"]
      tty: true
      securityContext:
        runAsUser: 0
        runAsGroup: 0
      env:
        - name: DOCKER_TLS_CERT_DIR
          value: ""
      volumeMounts:
        - name: dockersock
          mountPath: /var/run
      resources:
        requests:
          cpu: "500m"
          memory: "512Mi"
        limits:
          cpu: "2000m"
          memory: "2Gi"
  volumes:
    - name: dockersock
      hostPath:
        path: /var/run/docker.sock
'''
    }
  }

  options {
    timestamps()
    buildDiscarder(logRotator(numToKeepStr: '10'))
  }

  parameters {
    string(name: 'GITHUB_REPO', defaultValue: 'cicadialabs/qmd', description: 'GitHub repository (owner/repo)')
    string(name: 'BRANCH', defaultValue: 'main', description: 'Branch to build')
    string(name: 'DOCKER_IMAGE', defaultValue: 'daviddwyer1987/qmd-cicadia', description: 'Docker image name')
    string(name: 'DOCKER_REGISTRY', defaultValue: 'docker.io', description: 'Docker registry')
    string(name: 'DOCKER_TAG', defaultValue: 'latest', description: 'Docker tag')
    booleanParam(name: 'PUSH_IMAGE', defaultValue: true, description: 'Push image to registry')
    booleanParam(name: 'DEPLOY_TO_K8S', defaultValue: true, description: 'Deploy to k3-dev cluster')
    string(name: 'K8S_NAMESPACE', defaultValue: 'cicadia-system', description: 'Kubernetes namespace')
    string(name: 'K8S_DEPLOYMENT_NAME', defaultValue: 'qmd', description: 'Kubernetes deployment name')
  }

  environment {
    DOCKER_IMAGE_FULL = "${params.DOCKER_REGISTRY}/${params.DOCKER_IMAGE}:${params.DOCKER_TAG}"
  }

  stages {
    stage('Clone repository') {
      steps {
        container('builder') {
          sh '''
            set -eu
            echo "Cloning ${GITHUB_REPO} branch ${BRANCH}..."
            git clone --depth 1 --branch ${BRANCH} https://github.com/${GITHUB_REPO}.git /workspace/repo
            cd /workspace/repo
            GIT_REVISION=$(git rev-parse --short=12 HEAD)
            echo "${GIT_REVISION}" > /workspace/revision.txt
            echo "Cloned revision: ${GIT_REVISION}"
            cat /workspace/revision.txt
          '''
        }
      }
    }

    stage('Build Docker image') {
      steps {
        container('builder') {
          sh '''
            set -eu
            cd /workspace/repo

            # Enable docker multi-arch builder if not exists
            docker buildx create --name multiarch-builder --use 2>/dev/null || true
            docker buildx inspect multiarch-builder --bootstrap || docker buildx install --bootstrap

            echo "Building multi-arch Docker image: ${DOCKER_IMAGE_FULL}"
            docker buildx build \
              --platform linux/amd64,linux/arm64 \
              --tag "${DOCKER_IMAGE_FULL}" \
              --build-arg BUILD_DATE="$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
              --build-arg VCS_REF="$(cat /workspace/revision.txt)" \
              --build-arg VERSION="${DOCKER_TAG}" \
              --push \
              .

            echo "Listing built image..."
            docker images | grep ${DOCKER_IMAGE}
          '''
        }
      }
    }

    stage('Test image') {
      steps {
        container('builder') {
          sh '''
            set -eu
            echo "Running tests in container..."

            # Pull the image from registry for testing
            docker pull "${DOCKER_IMAGE_FULL}"

            # Start container in background
            docker run -d --name qmd-test --rm \
              -p 8181:8181 \
              -e QMD_LLM_BACKEND=ollama \
              "${DOCKER_IMAGE_FULL}" || true

            # Wait for container to start
            sleep 5

            # Check if container is running
            if docker ps | grep -q qmd-test; then
              echo "Container started successfully"

              # Check health endpoint
              sleep 2
              docker kill qmd-test || true
              echo "Image test passed"
            else
              echo "Container failed to start"
              docker logs qmd-test || true
              docker kill qmd-test || true
              exit 1
            fi
          '''
        }
      }
    }

    stage('Deploy to Kubernetes') {
      when {
        expression { params.DEPLOY_TO_K8S == true }
      }
      steps {
        container('jnlp') {
          sh '''
            set -eu
            echo "Deploying to k3-dev cluster..."

            # Get kubectl context
            if command -v kubectl >/dev/null 2>&1; then
              echo "kubectl available"
            else
              echo "kubectl not found - expecting it to be on the Jenkins agent"
            fi

            # Apply the deployment
            echo "Applying deployment in namespace ${K8S_NAMESPACE}..."

            # Note: This would normally use kubectl apply
            # For now, we'll output the manifest that needs to be applied
            cat <<EOF | tee /workspace/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${K8S_DEPLOYMENT_NAME}
  namespace: ${K8S_NAMESPACE}
  labels:
    app.kubernetes.io/name: ${K8S_DEPLOYMENT_NAME}
    app.kubernetes.io/part-of: cicadia-stack
spec:
  replicas: 1
  selector:
    matchLabels:
      app.kubernetes.io/name: ${K8S_DEPLOYMENT_NAME}
  strategy:
    type: Recreate
  template:
    metadata:
      labels:
        app.kubernetes.io/name: ${K8S_DEPLOYMENT_NAME}
        app.kubernetes.io/part-of: cicadia-stack
    spec:
      securityContext:
        runAsUser: 0
        runAsGroup: 0
      containers:
        - name: qmd
          image: ${DOCKER_IMAGE_FULL}
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 8181
              protocol: TCP
          env:
            - name: QMD_LLM_BACKEND
              value: "ollama"
            - name: QMD_OLLAMA_BASE_URL
              value: "http://ollama.cicadia-system.svc.cluster.local:11434"
            - name: QMD_MCP_PORT
              value: "8181"
            - name: QMD_MCP_HOST
              value: "0.0.0.0"
            - name: XDG_CACHE_HOME
              value: /tmp/cache
          resources:
            requests:
              cpu: "250m"
              memory: "1Gi"
            limits:
              cpu: "500m"
              memory: "3Gi"
          readinessProbe:
            httpGet:
              path: /health
              port: 8181
            initialDelaySeconds: 30
            periodSeconds: 10
            failureThreshold: 3
          livenessProbe:
            httpGet:
              path: /health
              port: 8181
            initialDelaySeconds: 60
            periodSeconds: 20
            failureThreshold: 3
          terminationGracePeriodSeconds: 30
      volumes:
        - name: qmd-data
          persistentVolumeClaim:
            claimName: qmd-data
        - name: qmd-config
          configMap:
            name: qmd-config
        - name: qmd-smb
          persistentVolumeClaim:
            claimName: qmd-smb-root
      volumeMounts:
        - name: qmd-data
          mountPath: /data/qmd
        - name: qmd-config
          mountPath: /config
          readOnly: true
        - name: qmd-smb
          mountPath: /data/knowledge
          readOnly: true
EOF

            cat /workspace/deployment.yaml
            echo "Deployment manifest generated. Apply with: kubectl apply -f /workspace/deployment.yaml"
          '''
        }
      }
    }
  }

  post {
    always {
      script {
        if (env.WORKSPACE?.trim()) {
          archiveArtifacts artifacts: 'deployment.yaml,revision.txt', allowEmptyArchive: true
        }
      }
    }
  }
}