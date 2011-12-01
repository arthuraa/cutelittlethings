(TeX-add-style-hook "git-cheat-sheet"
 (lambda ()
    (LaTeX-add-labels
     "sec:ssh")
    (TeX-add-symbols
     '("ind" 1)
     "bx"
     "E")
    (TeX-run-style-hooks
     "tikz"
     "enumerate"
     "array"
     "float"
     "graphicx"
     "amssymb"
     "amsmath"
     "geometry"
     "letterpaper"
     "inputenc"
     "latin9"
     "latex2e"
     "art12"
     "article"
     "12pt"
     "a4paper")))

