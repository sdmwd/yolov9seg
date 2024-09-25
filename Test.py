from PyQt5.QtWidgets import QListWidget, QListWidgetItem, QAbstractItemView

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
    self.load_image_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    button_layout.addWidget(self.load_image_button, 1)

    # Colonne de droite (liste des masques) - 1/4 de la largeur
    right_widget = QWidget()
    self.right_widget = right_widget
    right_widget.setMinimumWidth(int(self.width() * 0.25))
    right_layout = QVBoxLayout(right_widget)
    main_layout.addWidget(right_widget, 1)

    # Sub-layout for the two lists and arrow buttons
    list_layout = QHBoxLayout()
    right_layout.addLayout(list_layout)

    # Titre de la liste des masques segmentés
    mask_list_title = QLabel('Masques segmentés')
    list_layout.addWidget(mask_list_title)

    # Liste des masques segmentés (Enable multi-selection)
    self.mask_list = QListWidget()
    self.mask_list.setSortingEnabled(True)
    self.mask_list.setSelectionMode(QAbstractItemView.ExtendedSelection)  # Enable multi-selection
    self.mask_list.itemChanged.connect(self.toggle_mask_visibility)
    self.mask_list.itemDoubleClicked.connect(self.edit_mask_class)
    self.mask_list.itemClicked.connect(self.toggle_mask_selection)
    list_layout.addWidget(self.mask_list)

    # Arrow button layout (for moving masks between lists)
    arrow_button_layout = QVBoxLayout()

    # Add down arrow button for validating masks
    self.validate_arrow_button = QPushButton()
    self.validate_arrow_button.setIcon(QIcon("assets/down_arrow.png"))  # Use your own icon here
    self.validate_arrow_button.setToolTip("Valider le(s) masque(s)")
    self.validate_arrow_button.clicked.connect(self.validate_mask)
    arrow_button_layout.addWidget(self.validate_arrow_button)

    # Add up arrow button for unvalidating masks
    self.unvalidate_arrow_button = QPushButton()
    self.unvalidate_arrow_button.setIcon(QIcon("assets/up_arrow.png"))  # Use your own icon here
    self.unvalidate_arrow_button.setToolTip("Invalider le(s) masque(s)")
    self.unvalidate_arrow_button.clicked.connect(self.unvalidate_mask)
    arrow_button_layout.addWidget(self.unvalidate_arrow_button)

    # Add the arrow button layout between the two lists
    list_layout.addLayout(arrow_button_layout)

    # Titre de la liste des masques validés
    validated_list_title = QLabel('Masques validés')
    list_layout.addWidget(validated_list_title)

    # Liste des masques validés (Enable multi-selection)
    self.validated_mask_list = QListWidget()
    self.validated_mask_list.setSortingEnabled(True)
    self.validated_mask_list.setSelectionMode(QAbstractItemView.ExtendedSelection)  # Enable multi-selection
    self.validated_mask_list.itemChanged.connect(self.toggle_mask_visibility)
    list_layout.addWidget(self.validated_mask_list)

def validate_mask(self):
    """Moves selected masks from 'Masques segmentés' to 'Masques validés'."""
    selected_items = self.mask_list.selectedItems()
    for item in selected_items:
        self.mask_list.takeItem(self.mask_list.row(item))
        self.validated_mask_list.addItem(item)

def unvalidate_mask(self):
    """Moves selected masks from 'Masques validés' to 'Masques segmentés'."""
    selected_items = self.validated_mask_list.selectedItems()
    for item in selected_items:
        self.validated_mask_list.takeItem(self.validated_mask_list.row(item))
        self.mask_list.addItem(item)
