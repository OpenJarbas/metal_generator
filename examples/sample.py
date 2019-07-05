from metal_generator import MetalGenerator
from os.path import join, dirname

if __name__ == "__main__":
    name = "power_metal_bands"
    corpus = join(dirname(__file__), "metal_dataset", "{dataset}.txt".format(dataset=name))
    model_folder = join(dirname(__file__), "models")
    metal = MetalGenerator(corpus)
    metal.load(name, model_folder)
    print(metal.generate())
