import argparse
import os

from pydub import AudioSegment


def preprocess_songs(songs):
    """Preprocess the song files to make them the specified format.
    The songs used in the attack method must be a mono wav file with a sampling rate of 16kHz.

    Args:
        songs (str): Input directory of some song files.
    """
    song_filenames = os.listdir(songs)

    for song_filename in song_filenames:
        song = AudioSegment.from_file(file=os.path.join(songs, song_filename), format='wav')
        song = song.set_channels(1)
        song = song.set_frame_rate(16000)
        song.export(out_f=os.path.join(songs, song_filename), format='wav')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess the song files to make them the specified format.")
    parser.add_argument("-s", "--songs", type=str, default="./Audio Samples/Songs/", help="Input directory of some song files.")
    args = parser.parse_args()

    preprocess_songs(songs=args.songs)
