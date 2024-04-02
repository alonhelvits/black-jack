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
        for card in self.hand:
            if card in ['J', 'Q', 'K']:
                self.value += 10
            elif card == 'A':
                self.value += 11
                self.aces += 1
            else:
                self.value += int(card)
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


class GameState:
    def __init__(self):
        self.current_state = "betting"
        self.count_updated_this_round = False

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


def basic_strategy(player_hand, dealer_card):
    """Suggest the best action (hit or stand) based on the basic blackjack strategy."""
    player_value = sum([10 if card in ['J', 'Q', 'K'] else 11 if card == 'A' else int(card) for card in player_hand])
    player_aces = player_hand.count('A')
    dealer_value = 10 if dealer_card in ['J', 'Q', 'K'] else 11 if dealer_card == 'A' else int(dealer_card)

    # Adjust for Ace
    while player_value > 21 and player_aces:
        player_value -= 10
        player_aces -= 1

    # Simple Basic Strategy Logic
    if player_value >= 17:  # Stand on 17 or higher
        return 'Stand'
    elif player_value <= 11:  # Always hit 11 or less
        return 'Hit'
    elif player_value == 12 and dealer_value >= 4 and dealer_value <= 6:  # Stand if dealer shows 4-6, otherwise hit
        return 'Stand'
    elif 13 <= player_value <= 16 and dealer_value >= 2 and dealer_value <= 6:  # Stand if dealer shows 2-6, otherwise hit
        return 'Stand'
    else:
        return 'Hit'


def update_count(cards):
    global running_count
    count_values = {'2': 1, '3': 1, '4': 1, '5': 1, '6': 1, '7': 0, '8': 0, '9': 0, '10': -1, 'J': -1, 'Q': -1, 'K': -1,
                    'A': -1}
    for card in cards:
        running_count += count_values.get(card, 0)


def calculate_true_count():
    decks_remaining = max(decks_in_shoe - (len(detected_cards) / 52), 0.5)  # Avoid division by zero
    return running_count / decks_remaining


def bet_suggestion(true_count):
    if true_count <= 1.5:
        return "Bet the minimum"
    elif true_count <= 2.5:
        return "Bet double the minimum"
    elif true_count <= 3.5:
        return "Bet triple the minimum"
    else:
        return "Bet as much as you're comfortable with - the count is high!"


def print_results(dealer, players):
    print(f"Dealer's Hand: {dealer.hand}, Value: {dealer.value}")
    for player in players:
        result = ""
        if player.value > 21:
            result = "Busted"
        elif dealer.value > 21 or player.value > dealer.value:
            result = "Won"
        elif player.value == dealer.value:
            result = "Push"
        else:
            result = "Lost"
        print(f"Player's Hand: {player.hand}, Value: {player.value}, Result: {result}")


def process_game(dealer, players):
    global running_count, game_state_manager
    # dealer is an instance of the Dealer class, players is a list of Player instances
    # Detected no cards, potentially transitioning to the betting phase.
    if (not players or all(not player.hand for player in players)) and (not dealer or not dealer.hand):
        # Only transition to betting if previously in the result state, and reset count flag.
        if game_state_manager.is_result():
            game_state_manager.transition_to_betting()
            true_count = calculate_true_count()
            print("Running count is:", running_count)
            print("True count is:", true_count)
            print("detected cards:", detected_cards)
            print(bet_suggestion(true_count))
        elif game_state_manager.is_betting():
            # If already in betting phase, no need to reset the flag or transition.
            if running_count != 0:
                true_count = calculate_true_count()
                print(bet_suggestion(true_count))
        else:
            pass


    # Detected cards and potentially in the playing phase.
    elif len(dealer.hand) == 1:
        game_state_manager.transition_to_playing()
        dealer_card = dealer.hand[0]
        for player in players:
            action = basic_strategy(player.hand, dealer_card)
            print(f"Player's Hand: {player.hand}, Dealer's Card: {dealer_card}, Recommended Action: {action}")

    # Detected more than one card in the dealer's hand, potentially in the result phase.
    elif len(dealer.hand) > 1:
        dealer.calculate_value()
        # Ensure the transition to result state happens only once dealer's drawing is complete.
        if dealer.value >= 17:
            game_state_manager.transition_to_result()
            # Update the count only once after dealer finishes drawing and before the next round starts.
            if not game_state_manager.count_updated_this_round:
                all_cards = sum([player.hand for player in players], []) + dealer.hand
                update_count(all_cards)
                detected_cards.extend(all_cards)
                game_state_manager.count_updated_this_round = True  # Ensure count is updated only once per round.

            # Display results.
            print_results(dealer, players)
        else:
            print("Dealer still drawing cards..."")

    else:  # Handle transitional states or unexpected conditions. ???
        pass


# Card Counting Variables
running_count = 0
decks_in_shoe = 2  # Example for a game with two decks
game_state_manager = GameState()
dealer = Dealer()
players = [Player(), Player()]
detected_cards = []

# Assuming detected_cards is used within calculate_true_count and update_count,
# Initialize it as empty for the betting phase

process_game(dealer, players)
print("--------------------------------------------------")

dealer.set_hand(['K'])
players[0].set_hand(['5', '6'])
players[1].set_hand(['A', '2'])
process_game(dealer, players)
print("--------------------------------------------------")

dealer.set_hand(['K', '7'])  # Dealer now has cards sug
players[0].set_hand(['5', '6'])  # Players' hands remain unchanged from the playing phase
players[1].set_hand(['A', '2'])
process_game(dealer, players)
print("--------------------------------------------------")

dealer.set_hand([])  # Dealer now has cards sug
players[0].set_hand([])  # Players' hands remain unchanged from the playing phase
players[1].set_hand([])
process_game(dealer, players)
print("--------------------------------------------------")

dealer.set_hand(['8'])  # Dealer now has cards sug
players[0].set_hand(['5', '6'])  # Players' hands remain unchanged from the playing phase
players[1].set_hand(['A', '9'])
process_game(dealer, players)
print("--------------------------------------------------")

dealer.set_hand(['8'])  # Dealer now has cards sug
players[0].set_hand(['5', '6', '6'])  # Players' hands remain unchanged from the playing phase
players[1].set_hand(['A', '9'])
process_game(dealer, players)
print("--------------------------------------------------")

dealer.set_hand(['8', '7'])  # Dealer now has cards sug
players[0].set_hand(['5', '6', '6'])  # Players' hands remain unchanged from the playing phase
players[1].set_hand(['A', '9'])
process_game(dealer, players)
print("--------------------------------------------------")

dealer.set_hand(['8', '7', '5'])  # Dealer now has cards sug
players[0].set_hand(['5', '6', '6'])  # Players' hands remain unchanged from the playing phase
players[1].set_hand(['A', '9'])
process_game(dealer, players)
print("--------------------------------------------------")

dealer.set_hand([])  # Dealer now has cards sug
players[0].set_hand([])  # Players' hands remain unchanged from the playing phase
players[1].set_hand([])
process_game(dealer, players)
print("--------------------------------------------------")
