import numpy as np

class EditingDistance(object):
  """This class measures the editing distance between two sentences."""

  def __init__(self):
    super(EditingDistance, self).__init__()

  def _lcs(self, s1, s2):
    f = np.zeros((len(s1) + 1, len(s2) + 1), dtype='int')
    for i in range(len(s1)):
      for j in range(len(s2)):
        f[i + 1][j + 1] = max(f[i][j + 1], f[i + 1][j])
        if s1[i] == s2[j]:
          f[i + 1][j + 1] = max(f[i + 1][j + 1], f[i][j] + 1)
    return f[len(s1)][len(s2)]


  def __call__(self, s1, s2):
    s1 = s1.split()
    s2 = s2.split()
    if len(s1) == 0 or len(s2) == 0:
      return len(s1) + len(s2)
    return int(max(len(s1), len(s2)) - self._lcs(s1, s2))


if __name__ == "__main__":
  measure = EditingDistance()

  s1 = "aa bb cc ad cc da"
  s2 = "aa bb cc aa cc da"
  print(s1)
  print(s2)
  print(measure(s1, s2))


  s1 = "aa bb cc ad cc da"
  s2 = "aa bb aa cc da"
  print(s1)
  print(s2)
  print(measure(s1, s2))
