import timm
model_list = timm.list_models('*mobilenetv4*')
print(model_list)

model = timm.create_model('convnextv2_femto', pretrained=False)
print(f'Model size: {sum(p.numel() for p in model.parameters()) / 1e6} Million parameters')
