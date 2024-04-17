import sys, os

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_PATH)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from lipnet.generators import BasicGenerator
from lipnet.callbacks import Statistics, Visualize
from lipnet.curriculums import Curriculum
from lipnet.decoders import Decoder
from lipnet.helpers import labels_to_text
from lipnet.spell import Spell
from lipnet.model import LipNet
import numpy as np
import csv
import datetime
import tensorflow as tf

np.random.seed(55)

DATASET_DIR = os.path.join(CURRENT_PATH, 'datasets')
OUTPUT_DIR = os.path.join(CURRENT_PATH, 'results')
LOG_DIR = os.path.join(CURRENT_PATH, 'logs')

PREDICT_GREEDY = False
PREDICT_BEAM_WIDTH = 200
PREDICT_DICTIONARY = os.path.join(CURRENT_PATH, 'dictionaries', 'phrases.txt')


def curriculum_rules(epoch):
    return {'sentence_length': -1, 'flip_probability': 0.5, 'jitter_probability': 0.05}


# @tf.function
def train(run_name, start_epoch, stop_epoch, img_c, img_w, img_h, frames_n, absolute_max_string_len, minibatch_size,
          num_samples_stats):
    curriculum = Curriculum(curriculum_rules)
    lip_gen = BasicGenerator(dataset_path=DATASET_DIR,
                             minibatch_size=minibatch_size,
                             img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n,
                             absolute_max_string_len=absolute_max_string_len,
                             curriculum=curriculum, start_epoch=start_epoch, is_val=True).build()

    lipnet = LipNet(img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n,
                    absolute_max_string_len=absolute_max_string_len, output_size=lip_gen.get_output_size())
    lipnet.summary()

    # load weights
    #if start_epoch == 0:
        #start_file_w = os.path.join(OUTPUT_DIR, 'startWeight/V16_weights598.h5')
        #lipnet.model.load_weights(start_file_w)

    # load preexisting trained weights for the model weights113_peer_01.h5
    #if start_epoch > 0:
        #weight_file = os.path.join(OUTPUT_DIR, os.path.join(f"2023_11_06_17_35_37", f'weights%02d.h5' % (start_epoch - 1)))
        #lipnet.model.load_weights(weight_file)

    adam = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    spell = Spell(path=PREDICT_DICTIONARY)
    decoder = Decoder(greedy=PREDICT_GREEDY, beam_width=PREDICT_BEAM_WIDTH,
                      postprocessors=[labels_to_text, spell.sentence])


    statistics = Statistics(lipnet, lip_gen.next_val(), decoder, num_samples_stats,
                            output_dir=os.path.join(OUTPUT_DIR, run_name))
    visualize = Visualize(os.path.join(OUTPUT_DIR, run_name), lipnet, lip_gen.next_val(), decoder,
                          num_display_sentences=minibatch_size)

    statistics.on_train_begin()

    # Csv file to store train/val losses
    header_losses_csv = ["Epoch"] + ["ctc_loss"] + ["val_ctc_loss"]

    with open(os.path.join(LOG_DIR, f'training-{run_name}.csv'), 'w') as csvfile:
        csvw = csv.writer(csvfile)
        csvw.writerow(header_losses_csv)

    # Callback to initialize information for curriculum
    lip_gen.on_train_begin()

    for epoch in range(start_epoch, stop_epoch):

        # Store losses for each batch
        epoch_losses = []

        # callback to update curriculum rules
        lip_gen.on_epoch_begin(epoch)
        print("Epoch {}/{}".format(epoch, stop_epoch))

        # For each batch train simultaneously n students
        for batch in range(int(lip_gen.default_training_steps)):
            print("Batch {}/{}".format(batch, int(lip_gen.default_training_steps)))

            x_train, y_train = next(lip_gen.next_train())

            # train all students on the same batch
            with tf.GradientTape() as tape:
                model_out = lipnet.model(x_train, training=True)
                logits = model_out[0]
                losses = model_out[1]
                # Compute the mean CTC loss
                student_ctc_loss = tf.reduce_mean(losses, axis=0)
                print("Student mean loss: {}".format( student_ctc_loss))

                epoch_losses.append(student_ctc_loss)

            # Optimize students
            print("Optimizing student")
            gradients = tape.gradient(student_ctc_loss, lipnet.model.trainable_variables)
            adam.apply_gradients(zip(gradients, lipnet.model.trainable_variables))

        # Save weights for each student every epoch
        lipnet.model.save_weights(
                os.path.join(OUTPUT_DIR, run_name, "weights{:02d}.h5".format(epoch)))


        validation_loss = statistics.on_epoch_end(epoch)  # return mean val loss
        print(f"Validation loss: {validation_loss}")

        visualize.on_epoch_end(epoch)

        # Save logs
        # Training losses
        mean_epoch_losses = np.mean(np.array(epoch_losses), axis=0)

        # Merge Train and validation losses
        results = np.squeeze(np.stack((mean_epoch_losses, np.array(validation_loss).reshape(1))))

        with open(os.path.join(LOG_DIR, f'training-{run_name}.csv'), 'a') as csvfile:
            csvw = csv.writer(csvfile)
            row = [f"Epoch {epoch}"] + [results[0]] + [results[1]]
            csvw.writerow(row)


if __name__ == '__main__':
    run_name = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    start_epoch = 0
    stop_epoch = 600

    # Samples properties
    img_c = 3  # Image channels
    img_w = 100  # Image width
    img_h = 50  # Image height
    frames_n = 100  # Number of frames per video

    absolute_max_string_len = 54  # Max sentence length

    minibatch_size = 16  # Minibatch size

    num_samples_stats = 321  # Number of samples for statistics evaluation per epoch

    train(run_name, start_epoch, stop_epoch, img_c, img_w, img_h, frames_n, absolute_max_string_len, minibatch_size,
          num_samples_stats)


