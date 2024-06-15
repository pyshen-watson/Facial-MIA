import torch
from model.target import TargetModel, TargetType
from model.attack import AttackModel, AttackType


def convert_ONNX(model, dummy_input, export_path): 

    # Export the model   
    torch.onnx.export(
        model,         # model being run 
        dummy_input,       # model input (or a tuple for multiple inputs) 
        export_path,       # where to save the model  
        export_params=True,  # store the trained parameter weights inside the model file 
        opset_version=16,    # the ONNX version to export the model to 
        do_constant_folding=True,  # whether to execute constant folding for optimization 
        input_names = ['modelInput'],   # the model's input names 
        output_names = ['modelOutput'], # the model's output names 
        dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                    'modelOutput' : {0 : 'batch_size'}})
    
    print('Model has been converted to ONNX')
    

if __name__ == '__main__':
    
    target_model = TargetModel(TargetType.MBF_LARGE_V3).load('weights/target/mbf_large_v3.pt').backbone.eval()
    attack_model = AttackModel(AttackType.IDIAP).load('weights/attack/mbf_large_v3+idiap.pt').backbone.eval()

    dummy_image = torch.randn(1, 3, 112, 112)
    
    with torch.no_grad():
        dummy_feat = target_model(dummy_image)
        dummy_rec = attack_model(dummy_feat)

    convert_ONNX(target_model, dummy_image, 'weights/target/mbf_large_v3.onnx')
    convert_ONNX(attack_model, dummy_feat, 'weights/attack/mbf_large_v3+idiap.onnx')
    