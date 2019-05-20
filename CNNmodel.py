def build(width, height, depth,classesGenres,classesValence, param,activate="softmax"):

    regparam = param.regparam
    dropout = param.dropout

    inputShape = (height, width, depth)

    visible = Input(inputShape, name='input1')

    bn0=BatchNormalization(name='BN_input')(visible)
    conv1 = Conv2D(filters=64, kernel_size=2, strides=1, activation='relu',
                   kernel_initializer='glorot_normal',padding='same',name='Conv1')(bn0)
    bn1 = BatchNormalization(name='BN_conv1')(conv1)
    pool1 = MaxPooling2D(pool_size=(2,2), padding='same', name='pool1')(bn1)
    conv2 = Conv2D(filters=128, kernel_size=2, strides=1, activation='relu',
                   kernel_initializer='glorot_normal',padding='same', name='Conv2')(pool1)
    bn2 = BatchNormalization(name='BN_conv2')(conv2)
    pool2 = MaxPooling2D(pool_size=(2,2), padding='same', name='pool2')(bn2)
    conv3 =  Conv2D(filters=256, kernel_size=2, strides=1, activation='relu',
                   kernel_initializer='glorot_normal',padding='same', name='Conv3')(pool2)
    bn3 = BatchNormalization(name='BN_conv3')(conv3)
    pool3 = MaxPooling2D(pool_size=(2,2),padding='same', name='pool3')(bn3)

    conv4 =  Conv2D(filters=512, kernel_size=2, strides=2, activation='relu',
                   kernel_initializer='glorot_normal', padding='same', name='Conv4')(pool3)
    bn4 = BatchNormalization(name='BN_conv4')(conv4)
    pool4 = MaxPooling2D(pool_size=(2,2),padding='same', name='pool4Max')(bn4)
    pool5 = AveragePooling2D(pool_size=(2,2),padding='same', name='pool4Ave')(bn4)
    poolave = keras.layers.average([pool4,pool5], name='pool4Combined')

    flat = Flatten(name='Flatten')(poolave)
    dropd0=keras.layers.Dropout(dropout, name='Drop_Conv4')(flat)


    if param.FCLayers ==1:

        hidden1 = Dense(15, activation='relu',kernel_regularizer=regularizers.l2(regparam), name='FC1')(dropd0)
        bnd1 = BatchNormalization(name='BN_FC1')(hidden1)
        dropd1 = keras.layers.Dropout(dropout, name='DropOut_FC1')(bnd1)
        outputGenre = Dense(classesGenres, activation=activate, name="genreOutput")(dropd1)
        outputValence = Dense(classesValence, activation=activate, name = "valenceOutput")(dropd1)
    else:

        hidden1 = Dense(300, activation='elu',kernel_regularizer=regularizers.l2(regparam))(dropd0)
        bnd1 = BatchNormalization()(hidden1)
        dropd1 = keras.layers.Dropout(dropout)(bnd1)

        hidden2 = Dense(20, activation='elu',kernel_regularizer=regularizers.l2(regparam))(dropd1)
        bnd2 = BatchNormalization()(hidden2)
        dropd2 = keras.layers.Dropout(dropout)(bnd2)
        outputGenre = Dense(classesGenres, activation=activate, name="genreOutput")(dropd2)
        outputValence = Dense(classesValence, activation=activate, name = "valenceOutput")(dropd2)

    modelFunc = Model(inputs=visible, outputs=[outputGenre,outputValence])

    return modelFunc
