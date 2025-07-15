# 🎮 GridMaster AI

**GridMaster AI** is a constraint-based solver for *Ultimate Tic-Tac-Toe*, combining advanced techniques like CSP (Constraint Satisfaction Problem) solving, AC-3 arc consistency, forward checking, and minimax optimization. The goal: **outsmart your opponent** in the 3×3 grid of Tic-Tac-Toe boards — and win like a tactician.

---

## 📘 Game Overview: Ultimate Tic-Tac-Toe

Ultimate Tic-Tac-Toe is played on a 3×3 grid of smaller 3×3 Tic-Tac-Toe boards. The twist? Each move determines where your opponent plays next. To win, a player must win three small boards in a row on the large grid.

---

## 🧠 Task 1: Constraint Formulation

The game is formulated as a **Constraint Satisfaction Problem (CSP)** with:

### 🧩 Variables:
Each of the 81 positions across the 9 small boards is a variable:


### 📦 Domains:
Each variable can be:
- `'X'`
- `'O'`
- `Empty`

### 🔒 Constraints:

- **Move Legality**: Player must move in the small board indicated by the last move.
- **No Overwriting**: Cannot play in an occupied cell.
- **Win Constraints**:
  - Small board is won if a player aligns 3 marks.
  - No further moves allowed on a won board.
  - Game is won by claiming 3 small boards in a row.
- **Board Availability**: If the directed small board is full/won, the player may choose any available cell.

---

## 🤖 Task 2: CSP Solver Implementation

We use:

- ✅ **Backtracking Search**
- ✂️ **Forward Checking**
- 🔄 **Arc Consistency (AC-3)**
- 🧮 **Constraint Propagation**

### Key Features:

- Detects valid moves and prunes illegal ones.
- Prevents opponent wins using inference.
- Optimizes for quickest path to victory.
- Plays both offensive and defensive strategies.

---

## 🔬 Task 3: Experimentation & Analysis

### 🧪 Evaluation Methods:
- Play human vs AI
- Test various heuristics:
  - **MRV (Minimum Remaining Values)**
  - **Degree Heuristic**
  - **Most Constraining Variable**

### 🤖 vs 🤖:
- Compare CSP solver with:
  - Basic **Minimax**
  - **Minimax + Alpha-Beta Pruning**
  - **CSP + Alpha-Beta Hybrid**

### 📊 Results:
- CSP agent typically wins faster and more efficiently.
- Hybrid approach balances exploration and inference well.
- Alpha-Beta pruning helps reduce unnecessary searches.

---

## 🛠️ How to Run

### 🔧 Requirements

- Python 3.8+
- `numpy`
- `collections`
- `copy`

### ▶️ Run the Game

```bash
git clone https://github.com/your-username/gridmaster-ai.git
cd gridmaster-ai
python game.py
```

You can choose to:

Play against AI

Let AI play vs AI

Enable debug logs and board tracing
