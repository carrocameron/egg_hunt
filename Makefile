include .env

.PHONY: run-api docker-build-local docker-run docker-build-cloud docker-push docker-deploy

GCP_IMAGE = $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT_ID)/$(GCP_REPOSITORY)/$(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG)

run-api:
	uv run uvicorn api.fast:app --reload --port 8080

docker-build-local:
	docker build -t $(DOCKER_IMAGE_NAME):dev .

docker-run:
	docker run -it --rm -p 8080:8080 $(DOCKER_IMAGE_NAME):dev

docker-build-cloud:
	docker buildx build --platform linux/amd64 -t $(GCP_IMAGE) --load .

docker-push:
	docker push $(GCP_IMAGE)

# TensorFlow + model need more than Cloud Run's default 512 MiB
docker-deploy:
	gcloud run deploy $(GCP_SERVICE) \
	    --image $(GCP_IMAGE) \
	    --region $(GCP_REGION) \
	    --project $(GCP_PROJECT_ID) \
	    --port 8080 \
	    --memory 2Gi \
	    --allow-unauthenticated
