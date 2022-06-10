import falcon

from text_image_similarity import TextImageSimilarity

text_image_similarity_object = TextImageSimilarity()

# api = application = falcon.API()
# api.add_route("/text_image_similarity", text_image_similarity_object)
app = application = falcon.App()
app.add_route("/text_image_similarity", text_image_similarity_object)
