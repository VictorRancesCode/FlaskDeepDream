import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug import secure_filename
from flask import jsonify

# para deep dream
import numpy as np
from functools import partial
import PIL.Image
import tensorflow as tf
import urllib.request
import os
from PIL import Image

app = Flask(__name__, static_folder="generate")
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])


@app.route('/')
def index():
    return render_template('index.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


def GenerarFoto(filename, name, style):
    data_dir = './data/'
    img_noise = np.random.uniform(size=(224, 224, 3)) + 100.0

    model_fn = 'tensorflow_inception_graph.pb'

    graph = tf.Graph()
    sess = tf.InteractiveSession(graph=graph)
    with tf.gfile.FastGFile(os.path.join(data_dir, model_fn), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    t_input = tf.placeholder(np.float32, name='input')
    imagenet_mean = 117.0
    t_preprocessed = tf.expand_dims(t_input - imagenet_mean, 0)
    tf.import_graph_def(graph_def, {'input': t_preprocessed})
    layers = [op.name for op in graph.get_operations() if op.type == 'Conv2D' and 'import/' in op.name]
    feature_nums = [int(graph.get_tensor_by_name(name + ':0').get_shape()[-1]) for name in layers]

    print('Number of layers', len(layers))
    print('Total number of feature channels:', sum(feature_nums))

    def load_image_into_numpy_array(image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    def showarray(a, name):
        IMAGE_SIZE = (12, 8)
        a = np.uint8(np.clip(a, 0, 1) * 255)
        im = Image.fromarray(a)
        im.save('generate/' + name)
        print("paso")
        return None

    def T(layer):
        return graph.get_tensor_by_name("import/%s:0" % layer)

    def tffunc(*argtypes):
        placeholders = list(map(tf.placeholder, argtypes))

        def wrap(f):
            out = f(*placeholders)

            def wrapper(*args, **kw):
                return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))

            return wrapper

        return wrap

    def resize(img, size):
        img = tf.expand_dims(img, 0)
        return tf.image.resize_bilinear(img, size)[0, :, :, :]

    resize = tffunc(np.float32, np.int32)(resize)

    def calc_grad_tiled(img, t_grad, tile_size=512):
        sz = tile_size
        h, w = img.shape[:2]
        sx, sy = np.random.randint(sz, size=2)
        img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
        grad = np.zeros_like(img)
        for y in range(0, max(h - sz // 2, sz), sz):
            for x in range(0, max(w - sz // 2, sz), sz):
                sub = img_shift[y:y + sz, x:x + sz]
                g = sess.run(t_grad, {t_input: sub})
                grad[y:y + sz, x:x + sz] = g
        return np.roll(np.roll(grad, -sx, 1), -sy, 0)

    def render_deepdream(name, t_obj, img0=img_noise,
                         iter_n=10, step=1.5, octave_n=6, octave_scale=1.4):
        t_score = tf.reduce_mean(t_obj)
        t_grad = tf.gradients(t_score, t_input)[0]
        img = img0
        octaves = []
        for _ in range(octave_n - 1):
            hw = img.shape[:2]
            lo = resize(img, np.int32(np.float32(hw) / octave_scale))
            hi = img - resize(lo, hw)
            img = lo
            octaves.append(hi)
        for octave in range(octave_n):
            if octave > 0:
                hi = octaves[-octave]
                img = resize(img, hi.shape[:2]) + hi
            for _ in range(iter_n):
                g = calc_grad_tiled(img, t_grad)
                img += g * (step / (np.abs(g).mean() + 1e-7))
            showarray(img / 255.0, name)

    layer = 'mixed4d_3x3_bottleneck_pre_relu'
    channel = 139
    img0 = PIL.Image.open(filename)
    image_np = load_image_into_numpy_array(img0)
    render_deepdream(name, T(layer)[:, :, :, int(style)], image_np)
    return None


@app.route('/magia/', methods=['POST'])
def GenerarImagens():
    style = request.form['estilo']
    if int(style) <= 139:
        file = request.files['file']
        if file and allowed_file(file.filename):
            aux = file.filename
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            GenerarFoto(app.config['UPLOAD_FOLDER'] + filename, aux, style)
            return jsonify({"res": aux})
    return jsonify({"res": "error"})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
