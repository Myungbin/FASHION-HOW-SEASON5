import timm
model_list = timm.list_models('*edgenext_small.usi_in1k*', pretrained=True)
print(model_list)


model = timm.create_model("eva02_tiny_patch14_224.mim_in22k", pretrained=True)

print(model)


