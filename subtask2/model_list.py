import timm
model_list = timm.list_models('*eva02*')
print(model_list)

# model = timm.create_model('convnextv2_atto', pretrained=False)
# print(model)
