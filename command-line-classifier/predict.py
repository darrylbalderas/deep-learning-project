import argparse
import util

image_file = "./flowers/test/1/image_06743.jpg"

parser = argparse.ArgumentParser(description='Predict on an image using model checkpoint file')

parser.add_argument('image_path', nargs='?', type=util.check_file)
parser.add_argument('checkpoint_file', nargs='?')
parser.add_argument('--topk', action='store', default=3, type=int, help='top KK most likely classes')
parser.add_argument('--category_names', action='store', help='Mapping of categories to real names')
parser.add_argument('--gpu', action='store_true', default=False, help='Training via gpu')

args = parser.parse_args()
image_path = args.image_path
checkpoint_file = args.checkpoint_file
topk = args.topk
category_names = args.category_names
gpu = args.gpu

if image_path == None:
    raise parser.error("Image file path was not specified")
        
if checkpoint_file == None:
    raise parser.error("Model checkpoint file was not specified")
       
if not util.is_file(checkpoint_file) or not checkpoint_file.endswith(".pth"):
    raise parser.error("Make sure your checkpoint file exist and has a .pth extension")


model = util.load_model(checkpoint_file)
probs, labels = util.predict(image_path, model, topk, gpu)

if category_names != None:
    cat_to_names = util.get_category_names(category_names)
    util.show_flower_labels(probs, labels, cat_to_names)
else:
    util.show_flower_labels(probs, labels)
    