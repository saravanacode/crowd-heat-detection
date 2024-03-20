from moviepy.editor import VideoFileClip

clip = VideoFileClip("output.mp4")
clip.write_gif("output.gif")

