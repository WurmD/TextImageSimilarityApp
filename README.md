# ImageTextSimilarityApp

## Requirements

- docker (tested on docker 20.10.8 in Mac OS 11.4)

## To build

`docker build --tag text_image_similarity_image .`

## To run

- In one terminal:
`docker run --name text_image_similarity_container -p 8000:8000 -ti text_image_similarity_image:latest gunicorn text_image_similarity_app` which will start up the service, exposing http://127.0.0.1:8000/text_image_similarity as the API endpoint
- In another terminal:
`(echo -n '{"image": "'; base64 Djur_034.jpg; echo '"}') | curl -i -H "Content-Type: application/json" -d @- "http://127.0.0.1:8000/text_image_similarity?text=Blue%20insect%20on%20tree%20branch"`
to send image Djur_034.jpg and text 'Blue insect on tree branch' to the created endpoint http://127.0.0.1:8000/text_image_similarity