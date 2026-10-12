import zipfile
from fairmat_readers_xrd import read_rigaku_rasx
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

def find_intercept(df, peak_point, dir="both"):
  x, y = peak_point
  half_intensity = y / 2
  if dir == 'left':
    df_sel = df[df['2Theta'] < x]
    df_sel = df_sel.sort_values(by = '2Theta', ascending = False).reset_index(drop=True)
  elif dir == 'right':
    df_sel = df[df['2Theta'] > x]
    df_sel = df_sel.sort_values(by = '2Theta', ascending = True).reset_index(drop=True)
  else:
    return (find_intercept(df, peak_point, 'left'), find_intercept(df, peak_point, 'right'))
  df_sel = df_sel[['2Theta', 'Intensity']]
  x1, y1 = None, None
  x2, y2 = x, y
  for row in df_sel.itertuples():
    x1, y1 = x2, y2
    _, x2, y2 = row
    if y2 < half_intensity:
      break
  if x1 is not None:
    if y1 != y2:
      x_intercept = (x2-x1)/(y2-y1)*(half_intensity-y1) + x1
    else:
      x_intercept = (x2+x1)/2
  else:
    x_intercept = -1e12 if dir=='left' else 1e12
  return x_intercept

def load_rasx(rasx):
  # with zipfile.ZipFile(rasx, "r") as archive:
  #   with archive.open("Data0/Profile0.txt") as profile_file:
  #     df = pd.read_csv(
  #       profile_file, sep=r"\s+", names=["2Theta", "Intensity"]
  #     )
  data = read_rigaku_rasx(rasx)
  # Extract X (2-Theta) and Y (Intensity) coordinate arrays
  x_coords = data['2Theta']
  y_coords = data["intensity"]
  return pd.DataFrame({"2Theta": x_coords, "Intensity": y_coords})

def load_and_calc_q(rasx):
  df = load_rasx(rasx)
  fig, ax = plt.subplots()
  ax.plot(df['2Theta'], df['Intensity'])

  max_idx = df['Intensity'].idxmax()
  peak_intensity = df.iloc[max_idx]['Intensity']
  peak_point = (df.iloc[max_idx]['2Theta'], peak_intensity)
  left_x, right_x = find_intercept(df, peak_point)
  half_width = right_x - left_x

  segments = [
    [(left_x, peak_intensity/2), (right_x, peak_intensity/2)],
    [(left_x, peak_intensity/2), (left_x, 0)],
    [(right_x, peak_intensity/2), (right_x, 0)],
  ]
  line_segments = LineCollection(segments, colors='red', linestyles='-')
  ax.add_collection(line_segments)

  return {
    'peak_intensity': peak_intensity,
    'half_width': half_width,
    'q': peak_intensity / half_width,
    'fig': fig
  }

if __name__ == "__main__":
    calced = load_and_calc_q("MTV_MIL_160_R1.rasx")
    print(calced)
    calced['fig'].savefig("example_rasx.png")