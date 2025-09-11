from deepface import DeepFace

# Analyze emotions in the image
result = DeepFace.analyze(img_path="out3.jpg", actions=["emotion"])

print("DeepFace Analysis Result:")
print(result)
