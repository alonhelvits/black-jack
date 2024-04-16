
import cv2
import numpy as np

rank_templates = {
    'Ace': cv2.imread('Card_Imgs/Ace.jpg', cv2.IMREAD_GRAYSCALE),
    'Two': cv2.imread('Card_Imgs/Two.jpg', cv2.IMREAD_GRAYSCALE),
    'Three': cv2.imread('Card_Imgs/Three.jpg', cv2.IMREAD_GRAYSCALE),
    'Four': cv2.imread('Card_Imgs/Four.jpg', cv2.IMREAD_GRAYSCALE),
    'Five': cv2.imread('Card_Imgs/Five.jpg', cv2.IMREAD_GRAYSCALE),
    'Six': cv2.imread('Card_Imgs/Six.jpg', cv2.IMREAD_GRAYSCALE),
    'Seven': cv2.imread('Card_Imgs/Seven.jpg', cv2.IMREAD_GRAYSCALE),
    'Eight': cv2.imread('Card_Imgs/Eight.jpg', cv2.IMREAD_GRAYSCALE),
    'Nine': cv2.imread('Card_Imgs/Nine.jpg', cv2.IMREAD_GRAYSCALE),
    'Ten': cv2.imread('Card_Imgs/Ten.jpg', cv2.IMREAD_GRAYSCALE),
    'Jack': cv2.imread('Card_Imgs/Jack.jpg', cv2.IMREAD_GRAYSCALE),
    'Queen': cv2.imread('Card_Imgs/Queen.jpg', cv2.IMREAD_GRAYSCALE),
    'King': cv2.imread('Card_Imgs/King.jpg', cv2.IMREAD_GRAYSCALE),
}

suit_template = {
    'Diamonds': cv2.imread('Card_Imgs/Diamonds.jpg', cv2.IMREAD_GRAYSCALE),
    'Hearts': cv2.imread('Card_Imgs/Hearts.jpg', cv2.IMREAD_GRAYSCALE),
    'Clubs': cv2.imread('Card_Imgs/Clubs.jpg', cv2.IMREAD_GRAYSCALE),
    'Spades': cv2.imread('Card_Imgs/Spades.jpg', cv2.IMREAD_GRAYSCALE)
}

covered_template = cv2.imread('Card_Imgs/Covered.jpg', cv2.IMREAD_GRAYSCALE)




class Card:
    def __init__(self, corners, center, transpose_image, contour,card_width, card_height,rank_img, rank_img_trans,suit_img, warped):
        self.corners = corners
        self.center = center
        self.transpose_image = transpose_image
        self.contour = contour
        self.card_width = card_width
        self.card_height = card_height
        self.rank_img = rank_img
        self.rank_img_trans = rank_img_trans
        self.suit_img = suit_img
        self.warped = warped
        self.group = None
        self.rank = None
        self.suit = None

def find_cards(image):

    BKG_THRESH = 25

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.medianBlur(gray, 9)
    blurred = cv2.GaussianBlur(blurred, (5,5), 0)
    #blurred = cv2.bilateralFilter(gray, 11, 13, 13)

    # Perform adaptive thresholding
    bkg_level = np.bincount(image.ravel()).argmax()
    thresh_level = bkg_level + BKG_THRESH

    retval, thresh = cv2.threshold(blurred, thresh_level, 255, cv2.THRESH_BINARY)

    # Find contours
    #contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    index_sort = sorted(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]), reverse=True)
    if len(contours) == 0:
        return [], []

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    cards = []

    # Loop over the contours
    for contour in contours:
        area = cv2.contourArea(contour)

        # Filter contours based on area
        if 34000 < area < 50000 :
            # Approximate the contour to obtain the corners
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.01 * peri, True)

            # Check if the contour has four corners
            if len(approx) == 4:
                # Calculate the center of the card
                average = np.sum(np.float32(approx), axis=0) / len(np.float32(approx))
                cent_x = int(average[0][0])
                cent_y = int(average[0][1])
                center = np.array([cent_x, cent_y])

                # Find width and height of card's bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                card_width, card_height = w, h

                warped = flattener(image, np.float32(approx), card_width, card_height)

                covered_diff = int(np.sum(cv2.absdiff(covered_template, warped) / 255))

                if covered_diff < 25000: # check if the card is covered, adjust number for sensitivity
                    rank_img= "Covered"
                    rank_img_trans = "Covered"
                    cards.append(Card(np.float32(approx), center, warped, contour, card_width, card_height, rank_img, rank_img_trans, warped))
                else:
                    Qcorner_zoom_rank, Qcorner_zoom_suit = get_corner(warped)
                    rank_img = get_rank_suit(Qcorner_zoom_rank)
                    suit_img = get_rank_suit(Qcorner_zoom_suit)

                    #rank_img_trans = get_runk(np.rot90(warped,2))
                    rank_img_trans = rank_img
                    cards.append(Card(np.float32(approx), center, warped, contour,card_width, card_height,rank_img, rank_img_trans,suit_img, warped))
    return cards

def get_corner(warped):
    # Width and height of card corner, where rank and suit are
    CORNER_WIDTH = 110
    CORNER_HEIGHT = 124

    # Grab corner of warped card image and do a 5x zoom
    Qcorner = warped[8:CORNER_HEIGHT + 8, 0:CORNER_WIDTH]
    Qcorner_zoom = cv2.resize(Qcorner, (0, 0), fx=4, fy=4)
    cv2.imshow('Qcorner_zoom', Qcorner_zoom)
    cv2.waitKey(1)

    Qcorner_suit = warped[CORNER_HEIGHT + 4, 2*Qcorner + 4, 2 :CORNER_WIDTH]
    Qcorner_zoom_suit = cv2.resize(Qcorner_suit, (0, 0), fx=4, fy=4)
    cv2.imshow('Qcorner_zoom_suit', Qcorner_zoom_suit)
    cv2.waitKey(1)
    return Qcorner_zoom, Qcorner_zoom_suit

def get_rank_suit(Qcorner_zoom):
    # Dimensions of rank/suit train images
    WIDTH = 72
    HEIGHT = 127

    # Sample known white pixel intensity to determine good threshold level
    black_level, white_level, _, _ = cv2.minMaxLoc(Qcorner_zoom)
    thresh_level = (white_level + black_level) // 2 + black_level // 16 + 12
    if (thresh_level <= 0):
        thresh_level = 1
    retval, Qrank_inv = cv2.threshold(Qcorner_zoom, thresh_level, 255, cv2.THRESH_BINARY_INV)

    # Find rank contour and bounding rectangle, isolate and find largest contour
    Qrank_cnts, hier = cv2.findContours(Qrank_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    Qrank_cnts = sorted(Qrank_cnts, key=cv2.contourArea, reverse=True)

    # Find bounding rectangle for largest contour, use it to resize query rank
    # image to match dimensions of the train rank image
    if len(Qrank_cnts) != 0:
        x1, y1, w1, h1 = cv2.boundingRect(Qrank_cnts[0])
        Qrank_roi = Qrank_inv[y1:y1 + h1, x1:x1 + w1]
        Qrank_sized = cv2.resize(Qrank_roi, (WIDTH, HEIGHT), 0, 0)
        img = Qrank_sized
    else:
        img = ""
    cv2.imshow('rank_img', img)
    cv2.waitKey(1)
    return img

def classify_card_number(card_rank_image, rank_templates, warped):
    best_match_name = "Unknown"
    best_match_diff = 4000  # Initialize with a large value

    if type(card_rank_image) == str:
        best_match_name = "Covered"
    else:
        # Iterate over rank templates
        for rank, rank_template in rank_templates.items():
            # Resize the rank template to match the size of the card corner template
            resized_rank_template = cv2.resize(rank_template,
                                               (card_rank_image.shape[1], card_rank_image.shape[0]))

            # Calculate the absolute difference between card corner template and resized rank template
            diff_img = cv2.absdiff(card_rank_image, resized_rank_template)

            # Calculate the sum of absolute differences
            rank_diff = int(np.sum(diff_img) / 255)

            # Update best match if a better match is found
            if rank_diff < best_match_diff:
                best_match_diff = rank_diff
                best_match_name = rank
    '''
    if best_match_name == "Nine" or best_match_name == "Ten" or best_match_name == "Queen":
        queen_diff_1 = int(np.sum(cv2.absdiff(queen_template_1, warped) / 255))
        queen_diff_2 = int(np.sum(cv2.absdiff(queen_template_2, warped) / 255))
        queen_diff_3 = int(np.sum(cv2.absdiff(queen_template_3, warped) / 255))
        queen_diff_4 = int(np.sum(cv2.absdiff(queen_template_4, warped) / 255))
        if queen_diff_1 < 38000 or queen_diff_2 < 38000 or queen_diff_3 < 38000 or queen_diff_4 < 38000:
            best_match_name = "Queen"
    '''
    return best_match_name, best_match_diff

def classify_suit_number(card_suit_image, suit_templates):
    best_match_suit = "Unknown"
    best_match_diff = 400000  # Initialize with a large value

    # Iterate over rank templates
    for suit, suit_template in suit_templates.items():
        # Resize the suit template to match the size of the card corner template
        resized_suit_template = cv2.resize(suit_template,(card_suit_image.shape[1], card_suit_image.shape[0]))

        # Calculate the absolute difference between card corner template and resized suit template
        diff_img = cv2.absdiff(card_suit_image, resized_suit_template)

        # Calculate the sum of absolute differences
        suit_diff = int(np.sum(diff_img) / 255)

        # Update best match if a better match is found
        if suit_diff < best_match_diff:
            best_match_diff = suit_diff
            best_match_suit = suit
    return best_match_suit
# Function for grouping cards based on spatial proximity
def group_cards(cards, image):
    dealer_cards = []
    player1_cards = []
    player2_cards = []
    board_height, board_width = image.shape[:2]

    upper_third_height = board_height / 2.2

    for card in cards:
        if not isinstance(card, Card):
            continue
        card_center_y = card.center[1]
        card_center_x = card.center[0]

        # Check if the card is in the upper third of the image
        if card_center_y < upper_third_height:
            dealer_cards.append(card.rank)
        else:
            # Check if the card is on the left or right side of the image
            if card_center_x < board_width / 2:
                player1_cards.append(card.rank)
            else:
                player2_cards.append(card.rank)

    return dealer_cards, [player1_cards, player2_cards]


def flattener(image, pts, w, h):
    """Flattens an image of a card into a top-down 200x300 perspective.
    Returns the flattened, re-sized, grayed image."""

    temp_rect = np.zeros((4, 2), dtype="float32")

    s = np.sum(pts, axis=2)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    diff = np.diff(pts, axis=-1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    # Need to create an array listing points in order of
    # [top left, top right, bottom right, bottom left]
    # before doing the perspective transform

    if w <= 0.9 * h:  # If card is vertically oriented
        temp_rect[0] = tl
        temp_rect[1] = tr
        temp_rect[2] = br
        temp_rect[3] = bl

    if w >= 1.1 * h:  # If card is horizontally oriented
        temp_rect[0] = bl
        temp_rect[1] = tl
        temp_rect[2] = tr
        temp_rect[3] = br

    # If the card is 'diamond' oriented, a different algorithm
    # has to be used to identify which point is top left, top right
    # bottom left, and bottom right.

    if w > 0.9 * h and w < 1.1 * h:  # If card is diamond oriented
        # If furthest left point is higher than furthest right point,
        # card is tilted to the left.
        if pts[1][0][1] <= pts[3][0][1]:
            # If card is titled to the left, approxPolyDP returns points
            # in this order: top right, top left, bottom left, bottom right
            temp_rect[0] = pts[1][0]  # Top left
            temp_rect[1] = pts[0][0]  # Top right
            temp_rect[2] = pts[3][0]  # Bottom right
            temp_rect[3] = pts[2][0]  # Bottom left

        # If furthest left point is lower than furthest right point,
        # card is tilted to the right
        if pts[1][0][1] > pts[3][0][1]:
            # If card is titled to the right, approxPolyDP returns points
            # in this order: top left, bottom left, bottom right, top right
            temp_rect[0] = pts[0][0]  # Top left
            temp_rect[1] = pts[3][0]  # Top right
            temp_rect[2] = pts[2][0]  # Bottom right
            temp_rect[3] = pts[1][0]  # Bottom left

    maxWidth = 400
    maxHeight = 600

    # Create destination array, calculate perspective transform matrix,
    # and warp card image
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], np.float32)
    M = cv2.getPerspectiveTransform(temp_rect, dst)
    warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    warp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)

    return warp
def detect_cards(input_image):
    # Assume you have an image containing multiple cards called input_image
    #input_image = cv2.imread('test_image.png')

    # Find cards in the input image
    cards = find_cards(input_image)
    marked_frame = input_image.copy()

    # Iterate over each detected card
    for card in cards:
        # Classify the card number using the corner template
        if not isinstance(card, Card):
            continue
        best_match_name, best_match_diff = classify_card_number(card.rank_img,rank_templates, card.warped)
        best_match_name_trans, best_match_diff_trans = classify_card_number(card.rank_img_trans,rank_templates, card.warped)
        if best_match_diff < best_match_diff_trans:
            card.rank = best_match_name
        else:
            card.rank = best_match_name_trans

        card.suit = classify_suit_number(card.suit_img, suit_template)
        # Draw contours on the original image
        cv2.drawContours(marked_frame, [card.contour], -1, (255, 0, 0), 3)
        if card.rank == "Covered":
            # Outline
            cv2.putText(marked_frame, "Covered", (card.center[0] - 50, card.center[1]), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 0, 0), 16)
            # Main text
            cv2.putText(marked_frame, "Covered", (card.center[0] - 50, card.center[1]), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (255, 255, 255), 2)
        else:
            # Outline
            cv2.putText(marked_frame, f"{card.rank} \n of \n {card.suit}", (card.center[0] - 50, card.center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 16)
            # Main text
            cv2.putText(marked_frame, f"{card.rank} \n of \n {card.suit}", (card.center[0] - 50, card.center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    dealer_cards, player_cards = group_cards(cards, input_image)
    return cards, dealer_cards, player_cards, marked_frame

    # # Show the marked frame with contours
    # cv2.imshow('Marked Frame', marked_frame)
    # cv2.waitKey(1)
    # # Wait for a key press and close all windows
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
