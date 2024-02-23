import os
import sys

from PyQt5.QtGui import QIcon, QImageReader, QImage, QPixmap, QTransform
from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtWidgets import QWidget, QAction, QMenu, QMainWindow, QVBoxLayout, QLabel, QDockWidget
from PyQt5.QtWidgets import QListWidget, QScrollArea, QMessageBox, QFileDialog, QListWidgetItem, QApplication
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog
from PIL import Image as PilImage

from functools import partial

from toolBar import ToolBar
from canvas import Canvas
from lib import newIcon
from zoomWidget import ZoomWidget
from grab_cut import Grab_cut
from generate_single_line import Process
import cv2

__appname__ = 'BoulderBody'
defaultFilename = '.'

import os
from PyQt5.QtWidgets import (QApplication, QDialog, QVBoxLayout, QLineEdit,
                             QFileDialog, QPushButton, QLabel)

class CustomSaveDialog(QDialog):
    def __init__(self, parent=None, default_dir=None):
        super(CustomSaveDialog, self).__init__(parent)
        self.setWindowTitle('Save File')
        layout = QVBoxLayout(self)
        self.dirpath = None

        # Directory selection
        self.dirLineEdit = QLineEdit(self)
        self.browseButton = QPushButton('Browse...', self)
        self.dirLineEdit.setText(default_dir)
        self.browseButton.clicked.connect(self.browseDirectory)
        layout.addWidget(QLabel('选择储存地址:'))
        layout.addWidget(self.dirLineEdit)
        layout.addWidget(self.browseButton)

        # Additional input fields
        self.levelLineEdit = QLineEdit(self)
        layout.addWidget(QLabel('输入线路等级:'))
        layout.addWidget(self.levelLineEdit)
        
        self.colorLineEdit = QLineEdit(self)
        layout.addWidget(QLabel('输入线路颜色:'))
        layout.addWidget(self.colorLineEdit)

        # Save button
        self.saveButton = QPushButton('Save', self)
        self.saveButton.clicked.connect(self.saveFile)
        layout.addWidget(self.saveButton)

        self.filePath = ''

    def browseDirectory(self):
        dirPath = QFileDialog.getExistingDirectory(self, "Select Directory")
        if dirPath:
            self.dirLineEdit.setText(dirPath)

    def saveFile(self):
        dirPath = self.dirLineEdit.text().strip()
        level = self.levelLineEdit.text().strip()
        color = self.colorLineEdit.text().strip()
        if dirPath and level and color:
            # Formulate file name and save path
            i = 0
            while True:
                fileName = f"{color}_{level}_{i}"
                if fileName in os.listdir(dirPath):
                    i += 1
                else:
                    break
            self.filePath = os.path.join(dirPath, fileName)
            self.dirpath = dirPath
            # Here, instead of actually saving a file, we just accept the dialog
            # In a real application, you would save the file to `self.filePath`
            self.accept()
            
            
    def getFilePath(self):
        return self.filePath
    
    def getFileDirectory(self):
        return self.dirpath



class WindowMixin(object):

    def menu(self, title, actions=None):
        menu = self.menuBar().addMenu(title)
        if actions:
            addActions(menu, actions)
        return menu

    def toolbar(self, title, actions=None):
        toolbar = ToolBar(title)
        toolbar.setObjectName('{}ToolBar'.format(title))
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        if actions:
            addActions(toolbar, actions)
        self.addToolBar(Qt.LeftToolBarArea, toolbar)
        return toolbar

class ResizedQWidget(QWidget):
    def sizeHint(self):
        return QSize(100, 150)


def newAction(parent, text, slot=None, shortcut=None,
              tip=None, icon=None, checkable=False,
              enable=True):
    a = QAction(text, parent)
    if icon is not None:
        a.setIcon(QIcon(icon))
    if shortcut is not None:
        a.setShortcut(shortcut)
    if tip is not None:
        a.setToolTip(tip)
        a.setStatusTip(tip)
    if slot is not None:
        a.triggered.connect(slot)
    if checkable:
        a.setCheckable(True)
    a.setEnabled(enable)
    return a


def addActions(widget, actions):
    for action in actions:
        if action is None:
            widget.addSeparator()
        elif isinstance(action, QMenu):
            widget.addMenu(action)
        else:
            widget.addAction(action)


class struct(object):

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class MainWindow(QMainWindow, WindowMixin):
    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = list(range(3))

    def __init__(self, defaultFilename=None):
        super().__init__()
        # self.showFullScreen()  # This will display the window in full-screen mode
        self.dirty = True
        self.mImgList = []
        self.dirname = None
        self._beginner = True
        self.rotation_degree = 0  # Attribute to track rotation degree
        self.image_out_np = {}
        self.default_save_dir = None
        # Application state
        self.filePath = None
        self.mattingFile = None
        self.defaultWidth = 1000  # Default width when not in full screen
        self.defaultHeight = 1000  # Default height when not in full screen
        self.resize(self.defaultWidth, self.defaultHeight)
        # self.setGeometry(1000, 1000, self.defaultWidth, self.defaultHeight)
        listLayout = QVBoxLayout()
        listLayout.setContentsMargins(0, 0, 0, 0)
        matResultShow = ResizedQWidget()
        matResultShow.resize(150, 150)

        self.pic = QLabel(matResultShow)
        self.pic.resize(150, 150)
        self.pic.setGeometry(50, 20, 150, 150)
        self.shapes = {
            '裁剪边框': None,
            '起步点': [],
            '结束点': None
        }

        # self.pic.resize(matResultShow.width(), matResultShow.height())
        # self.pic.setScaledContents(True)

        matResultShow.setLayout(listLayout)
        # self.resultdock = QDockWidget('Result Image', self)
        # # self.resultdock.adjustSize()
        # self.resultdock.setObjectName('result')
        # self.resultdock.setWidget(matResultShow)
        # self.resultdock.resize(150, 150)

        self.fileListWidget = QListWidget()
        self.fileListWidget.itemDoubleClicked.connect(
            self.fileitemDoubleClicked)
        fileListLayout = QVBoxLayout()
        fileListLayout.setContentsMargins(0, 0, 0, 0)
        fileListLayout.addWidget(self.fileListWidget)
        fileListContainer = QWidget()
        fileListContainer.setLayout(fileListLayout)
        self.filedock = QDockWidget('文件列表', self)
        self.filedock.setObjectName('Files')
        self.filedock.setWidget(fileListContainer)

        self.zoomWidget = ZoomWidget()

        self.canvas = Canvas(parent=self)
        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(True)
        self.scrollBars = {
            Qt.Vertical: scroll.verticalScrollBar(),
            Qt.Horizontal: scroll.horizontalScrollBar()
        }
        self.scrollArea = scroll
        self.canvas.scrollRequest.connect(self.scrollRequest)
        self.scrollArea.setWidgetResizable(True)
        self.setCentralWidget(scroll)
        # self.addDockWidget(Qt.RightDockWidgetArea, self.resultdock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.filedock)
        self.filedock.setFeatures(QDockWidget.DockWidgetFloatable)

        self.dockFeatures = QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetFloatable
        # self.resultdock.setFeatures(
        #     self.resultdock.features() ^ self.dockFeatures)s
        

        # Actions
        action = partial(newAction, self)

        open_file = action('&导入单张图像', self.openFile, 'Ctrl+O', '导入单张图像')
        open_dir = action('&导入文件夹', self.openDir,
                          'Ctrl+D', '导入文件夹')
        # open_next_img = action('&Next Image', self.openNextImg,
        #                        'Ctrl+N', 'Open next image')
        # open_pre_img = action('&Previous Image', self.openPreImg,
        #                        'Ctrl+M', 'Open previous image')
        save = action('&储存结果', self.saveFile, 'Crl+S', '储存结果')
        matting = action('&完成标记\n生成标记图像', self.generate,
                         'e', '完成标记，生成标记图像')

        crop = action('&标记图像', self.createShapeCropping, 'C', '裁剪')
        confirm_crop = action('&确认', self.confirm_select, 'Crl+C', '确认')
        rotate = action('&旋转', self.rotateImage, 'Crl+R', '旋转')
        cancel = action('&取消上个标记', self.cancelSelect, 'Crl+Z', '取消上个标记')
        self.scalers = {
            self.FIT_WINDOW: self.scaleFitWindow,
            self.FIT_WIDTH: self.scaleFitWidth,
            # Set to one to scale to 100% when loading files.
            self.MANUAL_ZOOM: lambda: 1,
        }

        # store actions for further handling
        self.actions = struct(save=save, open_file=open_file,
                              open_dir=open_dir, rotate=rotate, cancel=cancel,
                              # open_next_img=open_next_img, open_pre_img=open_pre_img,
                            matting=matting, crop=crop, confirm_crop = confirm_crop)

        # Auto saving: enable auto saving if pressing next
        # self.autoSaving = QAction('Auto Saving', self)
        # self.autoSaving.setCheckable(True)
        # self.autoSaving.setChecked()

        # set toolbar
        self.tools = self.toolbar('Tools')
        self.actions.all = (save, open_file, open_dir, rotate, cancel,
                            # open_pre_img, open_next_img, 
                            matting, crop, confirm_crop)
        addActions(self.tools, self.actions.all)

        # set status
        self.statusBar().showMessage('{} started.'.format(__appname__))

    def okToContinue(self):
        if self.dirty:
            reply = QMessageBox.question(self, "Attention",
                                         "you have unsaved changes, proceed anyway?",
                                         QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            if reply == QMessageBox.Cancel:
                return False
            elif reply == QMessageBox.Yes:
                return self.fileSave
        return True

    def resetState(self):
        self.canvas.resetState()

    def errorMessage(self, title, message):
        return QMessageBox.critical(self, title,
                                    '<p><b>%s</b></p>%s' % (title, message))

    def beginner(self):
        return self._beginner

    def advanced(self):
        return not self.beginner()
    
    def confirm_select(self):
        def format_shape(s):
            return dict(line_color=s.line_color.getRgb(),
                        fill_color=s.fill_color.getRgb(),
                        points=[(p.x(), p.y()) for p in s.points])
        choice = self.showSingleChoiceDialog()
        if choice:
            print("User selected:", choice)
            if choice == '起步点':
                self.shapes[choice].append(format_shape(self.canvas.shapes[-1]))
            else:
                self.shapes[choice] = format_shape(self.canvas.shapes[-1])
        self.actions.save.setEnabled(True)
        self.actions.rotate.setEnabled(True)
        self.actions.matting.setEnabled(True)
        self.actions.crop.setEnabled(True)
        
    def showSingleChoiceDialog(self):
        options = ['起步点', '结束点', '裁剪边框']
        item, ok = QInputDialog.getItem(self, "选择边框功能", "选项", options, 0, False)
        if ok and item:
            return item
        return None

    def openFile(self, _value=False):
        path = os.path.dirname(self.filePath) if self.filePath else '.'
        formats = ['*.%s' % fmt.data().decode("ascii").lower()
                   for fmt in QImageReader.supportedImageFormats()]
        filters = "Image (%s)" % ' '.join(formats)
        filename = QFileDialog.getOpenFileName(
            self, '%s - Choose Image or Label file' % __appname__, path, filters)
        if filename:
            if isinstance(filename, (tuple, list)):
                filename = filename[0]
            self.loadFile(filename)

    def openDir(self, dirpath=None):
        defaultOpenDirPath = dirpath if dirpath else '.'
        targetDirPath = QFileDialog.getExistingDirectory(self,
                                                         '%s - Open Directory' % __appname__, defaultOpenDirPath,
                                                         QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)
        self.importDirImages(targetDirPath)

    def importDirImages(self, dirpath):
        self.fileListWidget.clear()
        self.mImgList = self.scanAllImages(dirpath)
        # self.openNextImg()
        for imgPath in self.mImgList:
            item = QListWidgetItem(imgPath)
            self.fileListWidget.addItem(item)

    def scanAllImages(self, folderPath):
        extensions = ['.%s' % fmt.data().decode("ascii").lower()
                      for fmt in QImageReader.supportedImageFormats()]
        imageList = []

        for root, dirs, files in os.walk(folderPath):
            for file in files:
                if file.lower().endswith(tuple(extensions)):
                    relativePath = os.path.join(root, file)
                    path = os.path.abspath(relativePath)
                    imageList.append(path)
        imageList.sort(key=lambda x: x.lower())
        return imageList

    def fileitemDoubleClicked(self, item=None):
        currIndex = self.mImgList.index(item.text())
        if currIndex < len(self.mImgList):
            filename = self.mImgList[currIndex]
            if filename:
                self.loadFile(filename)
                
    def cancelSelect(self):
        if len(self.canvas.shapes) > 1:
            self.canvas.shapes = self.canvas.shapes[:-1]
            self.canvas.update
                
    def rotateImage(self):
        if self.image:
            # Rotate the image
            transform = QTransform()
            transform.rotate(90)
            self.image = self.image.transformed(transform)
            
            # Update the canvas with the rotated image
            self.canvas.loadPixmap(QPixmap.fromImage(self.image))
            self.canvas.adjustSize()
            # self.scrollArea.adjustSize()
            # self.scrollArea.update()
            self.canvas.update()

            # # Update the stored image data (if you're storing the original image data)
            # self.imageData = self.image.bits().asstring(self.image.byteCount())
            self.rotation_degree += 90
            self.rotation_degree %= 360
            # Resize the canvas or the scroll area
            # self.image.adjustSize()  # If pic is a QLabel
              # If you are using a QScrollArea to display the image
            
    def loadFile(self, filePath=None):
        self.resetState()
        self.canvas.setEnabled(False)

        # highlight the file item
        if filePath and self.fileListWidget.count() > 0:
            index = self.mImgList.index(filePath)
            fileWidgetItem = self.fileListWidget.item(index)
            fileWidgetItem.setSelected(True)

        if filePath and os.path.exists(filePath):
            # Load image using Pillow to check for EXIF orientation
            pil_img = PilImage.open(filePath)
            exif_data = pil_img._getexif()
            orientation_tag = 274  # EXIF orientation tag
            if exif_data and orientation_tag in exif_data:
                orientation = exif_data[orientation_tag]
                rotation_degrees = {
                    3: 180,
                    6: 270,
                    8: 90
                }.get(orientation, 0)
                pil_img = pil_img.rotate(rotation_degrees, expand=True)
                # Convert the Pillow image back to QImage
                image = QImage(pil_img.tobytes(), pil_img.width, pil_img.height, QImage.Format_RGB888)#.rgbSwapped()
            else:
                # For non-JPEG images or JPEGs without orientation data
                image = QImage(filePath)

        if image.isNull():
            self.errorMessage(u'Error opening file',
                              u'<p>Make sure <i>%s</i> is a valid image file.' % filePath)
            self.status('Error reading %s' % filePath)
            return False
        self.status('Loaded %s' % os.path.basename(filePath))
        self.image = image
        self.filePath = filePath
        self.canvas.loadPixmap(QPixmap.fromImage(image))
        self.canvas.setEnabled(True)
        self.adjustScale(initial=True)
        self.paintCanvas()
        # self.toggleActions(True)

    def status(self, message, delay=5000):
        self.statusBar().showMessage(message, delay)

    def adjustScale(self, initial=False):
        value = self.scalers[self.FIT_WINDOW if initial else self.zoomMode]()
        self.zoomWidget.setValue(int(100 * value))

    def toggleFullScreen(self):
        if self.isFullScreen():
            self.showNormal()  # Exit full screen
            self.resize(self.defaultWidth, self.defaultHeight)  # Resize to default size
        else:
            self.showFullScreen()  # Enter full screen
            
    def scaleFitWindow(self):
        """Figure out the size of the pixmap in order to fit the main widget."""
        e = 2.0  # So that no scrollbars are generated.
        w1 = self.centralWidget().width() - e
        h1 = self.centralWidget().height() - e
        a1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.canvas.pixmap.width() - 0.0
        h2 = self.canvas.pixmap.height() - 0.0
        a2 = w2 / h2
        return w1 / w2 if a2 >= a1 else h1 / h2

    def scaleFitWidth(self):
        # The epsilon does not seem to work too well here.
        w = self.centralWidget().width() - 2.0
        return w / self.canvas.pixmap.width()

    def paintCanvas(self):
        assert not self.image.isNull(), "cannot paint null image"
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        self.canvas.adjustSize()
        self.canvas.update()
            
    def askForConfirmation(self, title, text):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Question)
        msg_box.setWindowTitle(title)
        msg_box.setText(text)
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg_box.setDefaultButton(QMessageBox.Yes)
        
        retval = msg_box.exec_()
        return retval == QMessageBox.Yes

    def createShapeCropping(self):
        assert self.beginner()
        confirmation = self.askForConfirmation('标记图像', "拖动鼠标绘画边框，绘画后点击确认（选择当前边框是起步点，结束点，裁剪）")
        if confirmation:
            self.canvas.setEditing(False)
            self.actions.save.setEnabled(False)
            self.actions.rotate.setEnabled(False)
            self.actions.matting.setEnabled(False)
            self.actions.crop.setEnabled(False)
            
                

    def toggleDrawMode(self, edit=True):
        self.canvas.setEditing(edit)
        self.actions.createMode.setEnabled(edit)
        self.actions.editMode.setEnabled(not edit)

    
    def generate(self):
        self.process = Process()
        self.image_out_np = self.process.process(self.filePath, self.shapes, self.rotation_degree)
        # self.showResultImg(self.image_out_np)
        self.actions.save.setEnabled(True)
        
    def grabcutMatting(self):

        if self.mattingFile is None:
            self.mattingFile = Grab_cut()

        def format_shape(s):
            return dict(line_color=s.line_color.getRgb(),
                        fill_color=s.fill_color.getRgb(),
                        points=[(p.x(), p.y()) for p in s.points])
        print(self.canvas.shapes)
        shape = format_shape(self.canvas.shapes[-1])
        self.image_out_np = self.mattingFile.image_matting(self.filePath,
                                                           shape, iteration=10)
        self.showResultImg(self.image_out_np)
        self.actions.save.setEnabled(True)

    # def showResultImg(self, image_np):
    #     # resize to pic
    #     factor = min(self.pic.width() /
    #                  image_np.shape[1], self.pic.height() / image_np.shape[0])
    #     image_np = cv2.resize(image_np, None, fx=factor,
    #                           fy=factor, interpolation=cv2.INTER_AREA)
    #     # image_np = cv2.resize((self.pic.height(), self.pic.width()))
    #     image = QImage(image_np, image_np.shape[1],
    #                    image_np.shape[0], QImage.Format_ARGB32)
    #     matImg = QPixmap(image)
    #     self.pic.setPixmap(matImg)

    def saveFile(self):
        self._saveFile(self.saveFileDialog())

    def _saveFile(self, saved_path):
        if saved_path:
            Process.resultSave(saved_path, self.image_out_np)
            self.setClean()
            self.statusBar().showMessage('Saved to  %s' % saved_path)
            self.statusBar().show()

    def saveFileDialog(self,):
        dialog = CustomSaveDialog(parent=self, default_dir = self.default_save_dir)
        if dialog.exec_() == QDialog.Accepted:
            filePath = dialog.getFilePath()
            file_dir = dialog.getFileDirectory()
            if file_dir is not None:
                self.default_save_dir = file_dir
            print("File path to save:", filePath)
            return filePath
        return ''

    def currentPath(self):
        return os.path.dirname(self.filePath) if self.filePath else '.'

    def setClean(self):
        self.dirty = False
        self.actions.save.setEnabled(False)
        self.actions.crop.setEnabled(True)

    def openNextImg():
        pass

    def openPreImg():
        pass

    def scrollRequest(self, delta, orientation):
        units = - delta / (8 * 15)
        bar = self.scrollBars[orientation]
        bar.setValue(bar.value() + bar.singleStep() * units)



def read(filename, default=None):
    try:
        with open(filename, 'rb') as f:
            return f.read()
    except Exception:
        return default


def get_main_app(argv=[]):
    app = QApplication(argv)
    app.setApplicationName(__appname__)
    app.setWindowIcon(newIcon("app"))
    ex = MainWindow()
    ex.show()
    return app, ex


def main(argv=[]):
    app, ex = get_main_app(argv)
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main(sys.argv))
