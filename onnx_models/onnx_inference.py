import onnxruntime

def load_model(model_path, gpu=False):
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if gpu else ['CPUExecutionProvider']
    session = onnxruntime.InferenceSession(model_path, providers=providers)
    output_names = [x.name for x in session.get_outputs()]
    meta = session.get_modelmeta().custom_metadata_map  # metadata
    if 'stride' in meta:
        stride, names = int(meta['stride']), eval(meta['names'])
    else:
        stride = 32
        names = {0: 'motor vehicle', 1: 'non-motor vehicle', 2: 'pedestrian', 3: 'red light', 4: 'yellow light', 5: 'green light', 6: 'off light'}
    return stride, names, session, output_names

def model_inf(img, session, output_names):
    # inference
    outputs = session.run(output_names, {session.get_inputs()[0].name: img})

    return outputs[0] if len(outputs) == 1 else [x for x in outputs]