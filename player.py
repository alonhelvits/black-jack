import cv2


class Participant:
    """Base class for both Players and the Dealer, to inherit common attributes and methods."""

    def __init__(self):
        self.hand = []
        self.value = 0
        self.aces = 0

    def set_hand(self, cards):
        """Set the entire hand at once and recalculate the value."""
        ##cards is a list of strings representing the cards in the hand, ex. ['1', 'K', 'A']"
        self.hand = cards
        self.calculate_value()

    def calculate_value(self):
        """Calculate the value of the hand, adjusting for aces as necessary."""
        self.value, self.aces = 0, 0
        card_values = {
            "One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5,
            "Six": 6, "Seven": 7, "Eight": 8, "Nine": 9, "Ten": 10,
            "Jack": 10, "Queen": 10, "King": 10, "Ace": 11
        }
        for card in self.hand:
            if card in card_values:
                self.value += card_values[card]
                if card == "Ace":
                    self.aces += 1

        self.adjust_for_aces()

    def adjust_for_aces(self):
        """Adjust the value of the hand if there are aces and the value is over 21."""
        while self.value > 21 and self.aces:
            self.value -= 10
            self.aces -= 1


class Dealer(Participant):
    """Represents the dealer in the game, with specific behaviors if needed in the future."""
    pass


class Player(Participant):
    """Represents a player in the game."""
    pass


def create_game(dealer_cards, players_cards):
    """
    Create a game setup with one dealer and two players.

    Parameters:
    - dealer_cards: List of strings representing the dealer's card values.
    - players_cards: List of two lists, each representing a player's card values.

    Returns:
    A tuple containing the dealer and the two players as objects.
    """

    # Create dealer instance and set hand
    dealer = Dealer()
    dealer.set_hand(dealer_cards)

    # Create player instances and set hands
    players = [Player() for _ in range(2)]  # Adjust if you need more or fewer players
    for player, cards in zip(players, players_cards):
        player.set_hand(cards)

    return dealer, players


class GameState:
    def __init__(self):
        self.current_state = "betting"
        self.count_updated_this_round = False
        self.current_cards = []

    # Simplified state transition methods; removed redundancy in resetting count_updated_this_round.
    def transition_to_betting(self):
        self.current_state = "betting"
        if self.count_updated_this_round == True:  # Ensure we reset the count flag here.
            self.count_updated_this_round = False

    def transition_to_playing(self):
        self.current_state = "playing"

    def transition_to_result(self):
        self.current_state = "result"

    # Removed unnecessary methods to focus on key state checks and updates.
    def is_betting(self):
        return self.current_state == "betting"

    def is_playing(self):
        return self.current_state == "playing"

    def is_result(self):
        return self.current_state == "result"


def basic_strategy(player, dealer):
    """Suggest the best action (hit or stand) based on the basic blackjack strategy."""

    # Adjust for Ace
    while player.value > 21 and player.aces:
        player.value -= 10
        player.aces -= 1

    # Simple Basic Strategy Logic
    if player.value > 21:  # Busted already
        return 'Busted..'
    elif player.value == 21:  # Always stand on 21
        return 'BlackJack!'
    elif 17 <= player.value <= 20:  # Stand on 17 or higher
        return 'Stand'
    elif player.value <= 11:  # Always hit 11 or less
        return 'Hit'
    elif player.value == 12 and 4 <= dealer.value <= 6:  # Stand if dealer shows 4-6, otherwise hit
        return 'Stand'
    elif 13 <= player.value <= 16 and 2 <= dealer.value <= 6:  # Stand if dealer shows 2-6, otherwise hit
        return 'Stand'
    else:
        return 'Hit'


def update_count(cards, running_count):
    count_values = {'One': 1, 'Two': 1, 'Three': 1, 'Four': 1, 'Five': 1, 'Six': 1, 'Seven': 0, 'Eight': 0, 'Nine': 0,
                    'Ten': -1, 'Jack': -1, 'Queen': -1, 'King': -1,
                    'Ace': -1}
    running_count = float(running_count)
    for card in cards:
        running_count += count_values.get(card, 0)
    return running_count


def calculate_true_count(decks_remaining, running_count):
    return format(float(running_count) / decks_remaining, '.1f')


def bet_suggestion(true_count):
    if float(true_count) <= 1.5:
        return "Bet the minimum"
    elif float(true_count) <= 2.5:
        return "Bet double the minimum"
    elif float(true_count) <= 3.5:
        return "Bet triple the minimum"
    else:
        return "Bet as much as you're comfortable with - the count is high!"


def game_results(dealer, player):
    result = ""
    if player.value > 21:
        result = "Busted"
    elif dealer.value > 21 or player.value > dealer.value:
        result = "Won"
    elif player.value == dealer.value:
        result = "Push"
    else:
        result = "Lost"
    return result


def process_game(dealer_cards, players_cards, input_image, running_count, true_count, game_state_manager,
                 decks_remaining):

    dealer, players = create_game(dealer_cards, players_cards)
    game_image = input_image.copy()

    # configure the text drawing parameters
    height, width = input_image.shape[:2]

    # Center of the image
    center_x = int(round(width / 2))
    center_y = int(round(height / 2))
    center_of_image = (center_x, center_y)

    # Close to the bottom edge, center of the left side
    bottom_left_center_x = (width * 0.1) - 150  # Assuming 10% from the left is "center" of the left side
    bottom_left_center_y = height * 0.9 + 70  # Close to the bottom edge
    bottom_left_center_position = (int(bottom_left_center_x), int(bottom_left_center_y))

    # Close to the bottom edge, center of the right side
    bottom_right_center_x = (width * 0.9) - 700  # Assuming 10% from the right edge
    bottom_right_center_y = height * 0.9 + 70  # Close to the bottom edge
    bottom_right_center_position = (int(bottom_right_center_x), int(bottom_right_center_y))

    # Center of x axis, 60% towards up of y axis
    center_x_up_x = width / 2
    center_x_up_y = height * 0.6  # 60% down from the top
    center_x_up_position = (center_x_up_x, center_x_up_y)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Phase handling logic

    if (not players or all(not player.hand for player in players)) and (not dealer or not dealer.hand):
        if game_state_manager.is_result():
            game_state_manager.transition_to_betting()
            true_count = calculate_true_count(decks_remaining, running_count)
            bet_suggestion_text = bet_suggestion(true_count)

            cv2.putText(game_image, "Betting Phase",
                        (center_of_image[0] - 300, center_of_image[1]), font, 3, (255, 255, 255), 25)
            cv2.putText(game_image, "Betting Phase",
                        (center_of_image[0] - 300, center_of_image[1]), font, 3, (0, 0, 0), 8)
            cv2.putText(game_image, f"Running count: {running_count}",
                        (center_of_image[0] - 270, center_of_image[1] + 90),
                        font, 2, (255, 255, 255), 25)
            cv2.putText(game_image, f"Running count: {running_count}",
                        (center_of_image[0] - 270, center_of_image[1] + 90),
                        font, 2, (0, 0, 0), 8)
            cv2.putText(game_image, f"True count: {true_count}",
                        (center_of_image[0] - 270, center_of_image[1] + 180),
                        font, 2, (255, 255, 255), 25)
            cv2.putText(game_image, f"True count: {true_count}",
                        (center_of_image[0] - 270, center_of_image[1] + 180),
                        font, 2, (0, 0, 0), 8)
            cv2.putText(game_image, f"Decks Remaining: {format(decks_remaining, '.1f')}",
                        (center_of_image[0] - 270, center_of_image[1] + 270),
                        font, 2, (255, 255, 255), 25)
            cv2.putText(game_image, f"Decks Remaining: {format(decks_remaining, '.1f')}",
                        (center_of_image[0] - 270, center_of_image[1] + 270),
                        font, 2, (0, 0, 0), 8)
            cv2.putText(game_image, f"Bet Suggestion: {bet_suggestion_text}",
                        (center_of_image[0] - 270, center_of_image[1] + 360),
                        font, 2, (255, 255, 255), 25)
            cv2.putText(game_image, f"Bet Suggestion: {bet_suggestion_text}",
                        (center_of_image[0] - 270, center_of_image[1] + 360),
                        font, 2, (0, 0, 0), 8)

        elif game_state_manager.is_betting():
            if running_count != 0:
                true_count = calculate_true_count(decks_remaining, running_count)
                bet_suggestion_text = bet_suggestion(true_count)

                cv2.putText(game_image, "Betting Phase",
                            (center_of_image[0] - 300, center_of_image[1]), font, 3, (255, 255, 255), 25)
                cv2.putText(game_image, "Betting Phase",
                            (center_of_image[0] - 300, center_of_image[1]), font, 3, (0, 0, 0), 8)
                cv2.putText(game_image, f"Running count: {running_count}",
                            (center_of_image[0] - 270, center_of_image[1] + 90),
                            font, 2, (255, 255, 255), 25)
                cv2.putText(game_image, f"Running count: {running_count}",
                            (center_of_image[0] - 270, center_of_image[1] + 90),
                            font, 2, (0, 0, 0), 8)
                cv2.putText(game_image, f"True count: {true_count}",
                            (center_of_image[0] - 270, center_of_image[1] + 180),
                            font, 2, (255, 255, 255), 25)
                cv2.putText(game_image, f"True count: {true_count}",
                            (center_of_image[0] - 270, center_of_image[1] + 180),
                            font, 2, (0, 0, 0), 8)
                cv2.putText(game_image, f"Decks Remaining: {format(decks_remaining, '.1f')}",
                            (center_of_image[0] - 270, center_of_image[1] + 270),
                            font, 2, (255, 255, 255), 25)
                cv2.putText(game_image, f"Decks Remaining: {format(decks_remaining, '.1f')}",
                            (center_of_image[0] - 270, center_of_image[1] + 270),
                            font, 2, (0, 0, 0), 8)
                cv2.putText(game_image, f"Bet Suggestion: {bet_suggestion_text}",
                            (center_of_image[0] - 270, center_of_image[1] + 360),
                            font, 2, (255, 255, 255), 25)
                cv2.putText(game_image, f"Bet Suggestion: {bet_suggestion_text}",
                            (center_of_image[0] - 270, center_of_image[1] + 360),
                            font, 2, (0, 0, 0), 8)
            else:
                cv2.putText(game_image, "Initial Betting Phase",
                            (center_of_image[0] - 500, center_of_image[1]), font, 3, (255, 255, 255), 25)
                cv2.putText(game_image, "Initial Betting Phase",
                            (center_of_image[0] - 500, center_of_image[1]), font, 3, (0, 0, 0), 8)
                cv2.putText(game_image, f"Running count: {running_count}",
                            (center_of_image[0] - 270, center_of_image[1] + 90),
                            font, 2, (255, 255, 255), 25)
                cv2.putText(game_image, f"Running count: {running_count}",
                            (center_of_image[0] - 270, center_of_image[1] + 90),
                            font, 2, (0, 0, 0), 8)

        # Additional conditions for betting phase can be added here

    elif "Covered" in dealer.hand and len(dealer.hand) == 2:
        dealer.calculate_value()
        players[0].calculate_value()
        players[1].calculate_value()
        game_state_manager.transition_to_playing()
        cv2.putText(game_image, "Playing Phase",
                    (center_of_image[0] - 300, center_of_image[1]), font, 3, (255, 255, 255), 25)
        cv2.putText(game_image, "Playing Phase",
                    (center_of_image[0] - 300, center_of_image[1]), font, 3, (0, 0, 0), 8)

        cv2.putText(game_image, f"Dealer's Card: {dealer.value}",
                    (center_of_image[0] - 250, center_of_image[1] - 150), font, 2, (255, 255, 255), 25)
        cv2.putText(game_image, f"Dealer's Card: {dealer.value}",
                    (center_of_image[0] - 250, center_of_image[1] - 150), font, 2, (0, 0, 0), 8)

        # calculate basic strategy for each player, and display the recommended action
        if players[0].hand:
            action = basic_strategy(players[0], dealer)
            if players[0].aces != 0 and players[0].value <= 21:
                player_text = f"Hand: {players[0].value} / {players[0].value - 10} , {action}"
            else:
                player_text = f"Hand: {players[0].value} , {action}"
            cv2.putText(game_image, player_text,
                        bottom_left_center_position, font, 2, (255, 255, 255), 25)
            cv2.putText(game_image, player_text,
                        bottom_left_center_position, font, 2, (0, 0, 0), 8)

        if players[1].hand:
            action = basic_strategy(players[1], dealer)
            if players[1].aces != 0 and players[1].value <= 21:
                player_text = f"Hand: {players[1].value} / {players[1].value - 10} , {action}"
            else:
                player_text = f"Hand: {players[1].value} , {action}"

            cv2.putText(game_image, player_text,
                        bottom_right_center_position, font, 2, (255, 255, 255), 25)
            cv2.putText(game_image, player_text,
                        bottom_right_center_position, font, 2, (0, 0, 0), 8)


    elif len(dealer.hand) >= 2 and "Covered" not in dealer.hand:
        dealer.calculate_value()
        if dealer.value >= 17:
            cv2.putText(game_image, f"Dealer's Hand: {dealer.value}",
                        (center_of_image[0] - 250, center_of_image[1] - 150), font, 2, (255, 255, 255), 25)
            cv2.putText(game_image, f"Dealer's Hand: {dealer.value}",
                        (center_of_image[0] - 250, center_of_image[1] - 150), font, 2, (0, 0, 0), 8)
            game_state_manager.transition_to_result()
            cv2.putText(game_image, "Result Phase",
                        (center_of_image[0] - 300, center_of_image[1]), font, 3, (255, 255, 255), 25)
            cv2.putText(game_image, "Result Phase",
                        (center_of_image[0] - 300, center_of_image[1]), font, 3, (0, 0, 0), 8)

            if players[0].hand:
                players[0].calculate_value()
                result = game_results(dealer, players[0])

                cv2.putText(game_image, f"Hand: {players[0].value}",
                            (bottom_left_center_position[0], bottom_left_center_position[1] - 90), font, 2, (255, 255, 255), 25)
                cv2.putText(game_image, f"Hand: {players[0].value}",
                            (bottom_left_center_position[0], bottom_left_center_position[1] - 90), font, 2, (0, 0, 0), 8)
                player_text = f"Result: {result}"
                cv2.putText(game_image, player_text,
                            bottom_left_center_position, font, 2, (255, 255, 255), 25)
                cv2.putText(game_image, player_text,
                            bottom_left_center_position, font, 2, (0, 0, 0), 8)

            if players[1].hand:
                players[1].calculate_value()
                result = game_results(dealer, players[1])

                cv2.putText(game_image, f"Hand: {players[1].value}",
                            (bottom_right_center_position[0], bottom_right_center_position[1] - 90), font, 2, (255, 255, 255), 25)
                cv2.putText(game_image, f"Hand: {players[1].value}",
                            (bottom_right_center_position[0], bottom_right_center_position[1] - 90), font, 2, (0, 0, 0), 8)
                player_text = f"Result: {result}"
                cv2.putText(game_image, player_text,
                            bottom_right_center_position, font, 2, (255, 255, 255), 25)
                cv2.putText(game_image, player_text,
                            bottom_right_center_position, font, 2, (0, 0, 0), 8)

            if not game_state_manager.count_updated_this_round:
                all_cards = sum([player.hand for player in players], []) + dealer.hand
                running_count = format(update_count(all_cards, running_count), '.1f')
                decks_remaining = max(decks_remaining - (len(all_cards) / 52), 0.5)
                game_state_manager.count_updated_this_round = True

        else:
            cv2.putText(game_image, "Dealer Drawing....",
                        (center_of_image[0] - 300, center_of_image[1]), font, 3, (255, 255, 255), 25)
            cv2.putText(game_image, "Dealer Drawing....",
                        (center_of_image[0] - 300, center_of_image[1]), font, 3, (0, 0, 0), 8)
            cv2.putText(game_image, f"Dealer's Hand: {dealer.value}",
                        (center_of_image[0] - 250, center_of_image[1] - 150), font, 2, (255, 255, 255), 25)
            cv2.putText(game_image, f"Dealer's Hand: {dealer.value}",
                        (center_of_image[0] - 250, center_of_image[1] - 150), font, 2, (0, 0, 0), 8)

    # Handle any other unexpected state
    else:
        pass

    return game_image, running_count, true_count, game_state_manager, decks_remaining
