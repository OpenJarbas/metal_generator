# metal_generator
generate metal band/song names, per genre!

- [metal_generator](#metal-generator)
- [pre trained models](#pre-trained-models)
  * [model](#model)
- [samples](#samples)
  * [heavy metal bands](#heavy-metal-bands)
  * [stoner metal bands](#stoner-metal-bands)
  * [black metal bands](#black-metal-bands)
  * [viking metal song names](#viking-metal-song-names)
- [Credits](#credits)

LSTM trained on [metal_dataset](https://github.com/JarbasAl/metal_dataset)

to train edit train.py to point to the filename.txt to be used

to sample edit sample.pt to choose a model


# pre trained models

pre trained models can be found in [metal_generator/models](./metal_generator/models)

band names:
- heavy metal bands
- stoner metal bands
- black metal bands

song names:
- viking metal songs


## model

epochs = 10

batch_size = 128


        model = Sequential()
        model.add(LSTM(128, input_shape=(maxlen, len(chars))))
        model.add(Dense(len(chars), activation='softmax'))
        
        optimizer = RMSprop(lr=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# samples

samples [here](./samples)

hand picked examples that i liked bellow


## heavy metal bands

        saxon king
        vendetta
        seventh sin
        iron head
        iron horse
        unholy cadaver
        
## stoner metal bands

        funeral horse
        backwoods payback
        conhaque stripper
        the walrus
        stoned cobra
        fuzzgod
        high ruler
        
## black metal bands

        sacradis
        manifestation
        dark requiem
        dark ritual
        daimonion
        lupine fall
        
## viking metal song names

        cold heart of winter
        cold winterland
        flight of ravens
        look skyward
        ...and beyond
        vinland
        a viking funeral
        in my darkest hour
        before the silence
        life and death
        sword in hand
        
        
# Credits

[metal-archives](https://www.metal-archives.com/) - the enclyclopedia of metal, where the dataset was downloaded from
[pythonic-metal](https://github.com/ijmbarr/pythonic-metal) - inspiration to gather the [metal_dataset](https://github.com/JarbasAl/metal_dataset)
[char-rbm](https://colinmorris.github.io/blog/dreaming-rbms) - inspiration, i previously used this for the same purpose
[keras examples](https://github.com/keras-team/keras/tree/master/examples) - i pretty much copied the lstm example over