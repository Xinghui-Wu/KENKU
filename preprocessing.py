import argparse
import os

from pydub import AudioSegment


def preprocess_songs(songs):
    """[summary]

    Args:
        songs ([type]): [description]
    """
    song_filenames = os.listdir(songs)

    for song_filename in song_filenames:
        song = AudioSegment.from_file(file=os.path.join(songs, song_filename), format='wav')
        song = song.set_channels(1)
        song = song.set_frame_rate(16000)
        song.export(out_f=os.path.join(songs, song_filename), format='wav')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-s", "--songs", type=str, default="./Audio Samples/Songs/", help="")
    args = parser.parse_args()

    preprocess_songs(songs=args.songs)
