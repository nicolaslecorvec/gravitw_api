#################### PACKAGE ACTIONS ###################
# ----------------------------------
#      GCP
# ----------------------------------

# path of the files to upload to gcp
LOCAL_PATH=XXX

# project id
PROJECT_ID=trans-crawler-365115

# bucket name
BUCKET_NAME=galleon-mri

# bucket directory in which to store the uploaded file (we choose to name this data as a convention)
BUCKET_FOLDER=data

# name for the uploaded file inside the bucket folder (here we choose to keep the name of the uploaded file)
# BUCKET_FILE_NAME=another_file_name_if_I_so_desire.csv
BUCKET_FILE_NAME=$(shell basename ${LOCAL_PATH})

REGION=europe-west1

DOCKER_IMAGE_NAME=gravitwave

# ----------------------------------
#      Uvicorn & Docker
# ----------------------------------

run_api:
	uvicorn api.simple:app --reload  # load web server with code autoreload

build_container:
	docker build -t eu.gcr.io/${PROJECT_ID}/${DOCKER_IMAGE_NAME} .

start_docker_locally:
	docker run -v=${HOME}/.config/gcloud:/root/.config/gcloud -e "PROJECT_ID=${PROJECT_ID}" -e PORT=8000 -p 8000:8000 eu.gcr.io/${PROJECT_ID}/${DOCKER_IMAGE_NAME}

push_to_gcr:
	docker push eu.gcr.io/${PROJECT_ID}/${DOCKER_IMAGE_NAME}

deploy_to_gcr:
	gcloud run deploy --image eu.gcr.io/${PROJECT_ID}/${DOCKER_IMAGE_NAME} --platform managed --region europe-west1 --memory 2Gi
