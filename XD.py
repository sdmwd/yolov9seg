from PyQt5.QtGui import QBrush, QPixmap
import os

def create_ui(self):
    """Crée l'interface utilisateur principale."""
    # Widget central
    central_widget = QWidget()
    self.setCentralWidget(central_widget)

    # Layout principal horizontal
    main_layout = QHBoxLayout(central_widget)

    # Colonne de gauche (affichage de l'image) - 3/4 de la largeur
    left_widget = QWidget()
    left_widget.setMinimumWidth(int(self.width() * 0.75))
    left_layout = QVBoxLayout(left_widget)
    main_layout.addWidget(left_widget, 3)

    # Boutons en haut
    button_layout = QHBoxLayout()
    left_layout.addLayout(button_layout)

    # Bouton "Charger l'image"
    self.load_image_button = QPushButton('Charger l\'image')
    self.load_image_button.clicked.connect(self.load_image)
    self.load_image_button.setSizePolicy(
        QSizePolicy.Expanding, QSizePolicy.Fixed)
    button_layout.addWidget(self.load_image_button, 1)

    # Création de la scène et vue graphique
    self.scene = QGraphicsScene()
    self.graphics_view = GraphicsView(self.scene, self)
    left_layout.addWidget(self.graphics_view)

    # Charger le logo de bienvenue comme image d'arrière-plan
    welcome_image_path = resource_path('assets/welcome_logo.png')  # Assurez-vous que cette image existe
    self.set_background_image(welcome_image_path)

def set_background_image(self, image_path):
    """Sets a background image in the scene."""
    if os.path.exists(image_path):
        background_pixmap = QPixmap(image_path)
        self.scene.clear()  # Clear any previous content
        self.scene.setBackgroundBrush(QBrush(background_pixmap.scaled(self.scene.width(), self.scene.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)))

def load_image(self, image_path=None):
    """Charge une image depuis un fichier ou via le glisser-déposer."""
    if not image_path:
        # Ouvre une boîte de dialogue pour sélectionner une image
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, 'Charger Image', resource_path('images'), 'Images (*.png *.jpg *.jpeg)', options=options
        )
        if not file_name:
            return
        image_path = file_name

    self.image_path = image_path
    cv_image = cv2.imread(self.image_path)
    if cv_image is None:
        QMessageBox.warning(self, 'Erreur', 'Impossible de charger l\'image.')
        return

    self.image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    self.display_image()

def display_image(self):
    """Affiche l'image chargée dans la zone graphique."""
    self.scene.clear()  # Clear the scene before displaying the new image
    pixmap = self.cvimg_to_pixmap(self.image)
    self.image_item = self.scene.addPixmap(pixmap)
    self.image_item.setZValue(0)
    self.graphics_view.resetTransform()
    self.graphics_view.scale_factor = 1.0
    self.graphics_view.fitInView(
        self.scene.itemsBoundingRect(), Qt.KeepAspectRatio
    )

def cvimg_to_pixmap(self, cv_img):
    """Convertit une image OpenCV en QPixmap."""
    height, width, channel = cv_img.shape
    bytes_per_line = 3 * width
    q_image = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(q_image)
