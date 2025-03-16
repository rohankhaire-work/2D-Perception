import sys
import torch
import onnx
from tool.darknet2pytorch import Darknet


def transform_to_onnx(cfgfile, weightfile, batch_size=1, onnx_file_name=None):
    model = Darknet(cfgfile)

    model.print_network()
    model.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    dynamic = False
    if batch_size <= 0:
        dynamic = True

    input_names = ["input"]
    output_names = ['boxes', 'confs']

    if dynamic:
        x = torch.randn((1, 3, model.height, model.width), requires_grad=True)
        if not onnx_file_name:
            onnx_file_name = "yolov4_-1_3_{}_{}_dynamic.onnx".format(
                model.height, model.width)
        dynamic_axes = {"input": {0: "batch_size"}, "boxes": {
            0: "batch_size"}, "confs": {0: "batch_size"}}
        # Export the model
        print('Export the onnx model ...')
        torch.onnx.export(model,
                          x,
                          onnx_file_name,
                          export_params=True,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=input_names, output_names=output_names,
                          dynamic_axes=dynamic_axes)

        print('Onnx model exporting done')
        return onnx_file_name

    else:
        x = torch.randn((batch_size, 3, model.height,
                        model.width), requires_grad=True)
        onnx_file_name = args.onnx_file_path
        torch.onnx.export(model,
                          x,
                          onnx_file_name,
                          export_params=True,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=input_names, output_names=output_names,
                          dynamic_axes=None)

        print('Onnx model exporting done')

        if args.simplify:
            try:
                import onnxsim
                onnx_model = onnx.load(args.onnx_file_path)
                print('\nStarting to simplify ONNX...')
                onnx_model_simple, check = onnxsim.simplify(onnx_model)
                assert check, "Simplified ONNX model could not be validated"
            except Exception as e:
                print(f'Simplifier failure: {e}')

        # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        onnx.save(onnx_model_simple, args.onnx_file_path)
        print('ONNX export success, saved as %s' % args.onnx_file_path)

        return onnx_file_name


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('weightfile')
    parser.add_argument('--batch_size', type=int,
                        help="Static Batchsize of the model. use batch_size<=0 for dynamic batch size")
    parser.add_argument('--onnx_file_path', help="Output onnx file path")
    parser.add_argument(
        '--simplify', help="simplify using onnxsim", action='store_true')
    args = parser.parse_args()
    transform_to_onnx(args.config, args.weightfile,
                      args.batch_size, args.onnx_file_path)
