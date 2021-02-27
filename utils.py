from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string("--style_image", None, "path to the style image")
flags.DEFINE_string("--content_image", None, "path to the content image")

# Required flag.
flags.mark_flag_as_required("style_image")


def load_sc_images():
    pass

def setup_gen_image():
    pass
