"""

"""

import sys

from PyQt5.QtWidgets import QApplication, QWidget


class Window(QWidget):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        self.setWindowTitle("Hello World")
        self.resize(250, 300)



def main():
    root_ui = QApplication([])
    app = Window()
    app.show()



    sys.exit(root_ui.exec_())


if __name__ == "__main__":
    main()
