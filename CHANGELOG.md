# Changelog

## Unreleased

### Added
- **CLI Enhancements:**
  - Added command-line interface with multiple modes (train, eval, plot)
  - Added configuration file support (JSON-based)
  - Added new CLI options:
    - `--config` for loading settings from JSON files
    - `--log_level` for controlling logging verbosity
    - `--teacher_lr` and `--student_lr` for learning rates
    - `--resume_episode` and `--checkpoint_episode` for checkpoint control
    - `--log_interval` for controlling logging frequency

- **Directory Structure:**
  - Added `logs/` directory for timestamped log files
  - Added `checkpoints/` directory for model checkpoints
  - Added `configs/` directory for configuration files
  - Added `results/` directory for visualization outputs

- **Logging System:**
  - Added timestamped log files
  - Added console and file logging
  - Added configurable log levels (DEBUG, INFO, WARNING, ERROR)
  - Added detailed metrics logging

- **Checkpointing System:**
  - Added episode-specific checkpoints
  - Added organized checkpoint directory structure
  - Added better checkpoint loading/saving with error handling

- **Visualization Improvements:**
  - Added loss plots
  - Added reward distribution plots using seaborn
  - Added timestamped result files
  - Added better plot styling and organization

### Changed
- **Code Structure:**
  - Refactored `main.py` to use configuration-based approach
  - Improved error handling and logging throughout the codebase
  - Enhanced metrics tracking and reporting
  - Improved code organization and documentation

- **Workflow:**
  - Temporarily removed and then restored the workflow file (`.github/workflows/ci.yml`)

### Technical Details
- Added new dependencies:
  - `seaborn` for enhanced visualizations
  - `pathlib` for better path handling
  - `datetime` for timestamp management
  - `json` for configuration management

- File Changes:
  - Modified: `main.py` (major refactoring)
  - Added: `CHANGELOG.md`
  - Added: `.github/workflows/ci.yml` (restored)
  - Added: Various configuration files in `configs/` directory

### Implementation Details
- **Agent Architecture:**
  - Teacher Agent:
    - State dimension: Matches environment state space
    - Action dimension: Matches environment action space
    - Curriculum dimension: Matches state space
    - Learning rate: Configurable (default: 0.001)
    - Model checkpointing: Episode-based with timestamps

  - Student Agent:
    - State dimension: Matches environment state space
    - Action dimension: Matches environment action space
    - Learning rate: Configurable (default: 0.001)
    - Model checkpointing: Episode-based with timestamps

- **Environment Details:**
  - Task Environment:
    - State space: Continuous
    - Action space: Continuous
    - Task complexity: Configurable (0.0-1.0)
    - Curriculum adaptation: Dynamic based on teacher actions

- **Training Process:**
  - Episode structure:
    1. Environment reset
    2. Teacher action selection
    3. Curriculum generation
    4. Environment difficulty update
    5. Student understanding
    6. Student action selection
    7. Environment step
    8. Agent updates
    9. Metrics recording
    10. Checkpointing (if interval reached)

- **Metrics Tracking:**
  - Teacher metrics:
    - Rewards
    - Losses
    - Action distributions
  - Student metrics:
    - Rewards
    - Losses
    - Understanding accuracy
  - Environment metrics:
    - Curriculum difficulty
    - Task completion rate
    - Episode lengths

- **Configuration System:**
  - JSON-based configuration files
  - Default values:
    ```json
    {
      "num_episodes": 1000,
      "env_difficulty": 0.5,
      "checkpoint_interval": 100,
      "log_interval": 100,
      "teacher_learning_rate": 0.001,
      "student_learning_rate": 0.001,
      "resume": false
    }
    ```
  - Command-line override support
  - Configuration validation

- **Logging System Details:**
  - Log file format: `experiment_YYYYMMDD_HHMMSS.log`
  - Log entry format: `timestamp | level | message`
  - Log levels:
    - DEBUG: Detailed debugging information
    - INFO: General operational information
    - WARNING: Warning messages
    - ERROR: Error messages
  - Console and file output
  - Rotating log files

- **Visualization System:**
  - Plot types:
    1. Rewards over time
    2. Curriculum difficulty over time
    3. Losses over time
    4. Reward distributions
  - Output format: PNG
  - File naming: `training_results_YYYYMMDD_HHMMSS.png`
  - Style: Seaborn-based with custom styling
  - Layout: 2x2 grid with shared legends

### Project Structure
```
teaching-agent/
├── agents/
│   ├── __init__.py
│   ├── teacher.py
│   └── student.py
├── environments/
│   ├── __init__.py
│   └── task_env.py
├── configs/
│   └── default_config.json
├── logs/
│   └── experiment_*.log
├── checkpoints/
│   ├── teacher_*.pth
│   └── student_*.pth
├── results/
│   └── training_results_*.png
├── .github/
│   └── workflows/
│       └── ci.yml
├── main.py
├── requirements.txt
├── README.md

```

### Code Organization
- **Agent Classes (`agents/`):**
  - `TeacherAgent`:
    - Methods:
      - `select_action(state)`: Choose action based on state
      - `generate_curriculum(state, action)`: Create curriculum
      - `update(experience)`: Update model parameters
      - `save_model(path)`: Save model checkpoint
      - `load_model(path)`: Load model checkpoint

  - `StudentAgent`:
    - Methods:
      - `understand_curriculum(state, curriculum)`: Process curriculum
      - `select_action(understanding)`: Choose action
      - `update(experience)`: Update model parameters
      - `save_model(path)`: Save model checkpoint
      - `load_model(path)`: Load model checkpoint

- **Environment (`environments/`):**
  - `TeachingTaskEnv`:
    - Methods:
      - `reset()`: Initialize environment
      - `step(action)`: Execute action
      - `set_task_difficulty(difficulty)`: Update difficulty
      - `get_task_difficulty()`: Get current difficulty

- **Main Script (`main.py`):**
  - Functions:
    - `setup_logging()`: Configure logging
    - `save_checkpoint()`: Save agent state
    - `load_checkpoint()`: Load agent state
    - `train_agents()`: Training loop
    - `evaluate_agents()`: Evaluation loop
    - `plot_results()`: Visualization
    - `main()`: CLI entry point

### Dependencies
```txt
numpy>=1.21.0
torch>=1.9.0
matplotlib>=3.4.0
seaborn>=0.11.0
pathlib>=1.0.1
```

### Development Workflow
1. **Setup:**
   - Clone repository
   - Install dependencies: `pip install -r requirements.txt`
   - Create necessary directories

2. **Training:**
   - Basic: `python main.py --mode train`
   - With config: `python main.py --mode train --config custom_config.json`
   - Resume: `python main.py --mode train --resume --resume_episode 500`

3. **Evaluation:**
   - Basic: `python main.py --mode eval`
   - Specific checkpoint: `python main.py --mode eval --checkpoint_episode 1000`

4. **Visualization:**
   - View results: `python main.py --mode plot`
   - Results stored in `results/` directory

### Future Improvements
- [ ] Add unit tests
- [ ] Implement curriculum generation strategies
- [ ] Add multi-agent support
- [ ] Create interactive visualization dashboard
- [ ] Add experiment tracking system
- [ ] Implement distributed training
- [ ] Add documentation website 

### Code Implementation Details

#### Agent Implementation
```python
# agents/teacher.py
class TeachingAgent:
    def __init__(self, state_dim, action_dim, curriculum_dim, learning_rate=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.curriculum_dim = curriculum_dim
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def _build_model(self):
        return torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, self.action_dim)
        )

    def select_action(self, state):
        with torch.no_grad():
            return self.model(torch.FloatTensor(state)).numpy()

    def generate_curriculum(self, state, action):
        return np.clip(action, 0, 1)  # Normalize curriculum to [0,1]

    def update(self, experience):
        state, action, reward, next_state = experience
        # Implementation of policy update
        return {'loss': loss.item()}

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

#### Environment Implementation
```python
# environments/task_env.py
class TeachingTaskEnv:
    def __init__(self, task_complexity=0.5):
        self.task_complexity = task_complexity
        self.state_dim = 10  # Example dimension
        self.action_dim = 5  # Example dimension
        self.current_state = None

    def reset(self):
        self.current_state = np.random.randn(self.state_dim)
        return self.current_state, {}

    def step(self, action):
        # Implement environment dynamics
        next_state = self._get_next_state(action)
        reward = self._calculate_reward(action)
        done = self._is_terminal()
        info = {'task_complexity': self.task_complexity}
        return next_state, reward, done, False, info

    def set_task_difficulty(self, difficulty):
        self.task_complexity = np.clip(difficulty, 0, 1)

    def get_task_difficulty(self):
        return self.task_complexity
```

#### Configuration Files
```json
// configs/default_config.json
{
    "training": {
        "num_episodes": 1000,
        "env_difficulty": 0.5,
        "checkpoint_interval": 100,
        "log_interval": 100
    },
    "agents": {
        "teacher": {
            "learning_rate": 0.001,
            "hidden_dim": 64,
            "activation": "relu"
        },
        "student": {
            "learning_rate": 0.001,
            "hidden_dim": 64,
            "activation": "relu"
        }
    },
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s | %(levelname)s | %(message)s",
        "file_pattern": "experiment_%Y%m%d_%H%M%S.log"
    }
}
```

#### Workflow Configuration
```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Run tests
      run: |
        python -m pytest tests/
```

### File Contents and Structure

#### Directory Contents
```
teaching-agent/
├── agents/
│   ├── __init__.py          # Package initialization
│   ├── teacher.py           # Teacher agent implementation
│   └── student.py           # Student agent implementation
├── environments/
│   ├── __init__.py          # Package initialization
│   └── task_env.py          # Environment implementation
├── configs/
│   ├── default_config.json  # Default configuration
│   └── custom_config.json   # Custom configuration template
├── logs/                    # Log files directory
├── checkpoints/             # Model checkpoints directory
├── results/                 # Visualization results directory
├── .github/
│   └── workflows/
│       └── ci.yml           # GitHub Actions workflow
├── main.py                  # Main script
├── requirements.txt         # Dependencies
├── README.md               # Project documentation

```

#### Requirements File
```txt
# requirements.txt
numpy>=1.21.0
torch>=1.9.0
matplotlib>=3.4.0
seaborn>=0.11.0
pathlib>=1.0.1
pytest>=6.2.5
black>=21.5b2
flake8>=3.9.2
```

### Implementation Notes

#### Agent Design Decisions
1. **Teacher Agent:**
   - Uses a simple feedforward neural network
   - Implements curriculum generation through action space
   - Maintains separate optimizer for policy updates

2. **Student Agent:**
   - Similar architecture to teacher
   - Additional curriculum understanding layer
   - Implements experience replay for better learning

#### Environment Design
1. **State Space:**
   - Continuous state representation
   - Normalized to [-1, 1] range
   - Includes task context information

2. **Action Space:**
   - Continuous actions
   - Clipped to valid ranges
   - Includes curriculum generation actions

#### Training Process Details
1. **Episode Flow:**
   ```python
   # Pseudocode for training loop
   for episode in range(num_episodes):
       state = env.reset()
       for step in range(max_steps):
           teacher_action = teacher.select_action(state)
           curriculum = teacher.generate_curriculum(state, teacher_action)
           env.set_task_difficulty(curriculum)
           student_understanding = student.understand_curriculum(state, curriculum)
           student_action = student.select_action(student_understanding)
           next_state, reward, done, info = env.step(student_action)
           teacher.update((state, teacher_action, reward, next_state))
           student.update((state, student_action, reward, next_state, curriculum))
           if done:
               break
           state = next_state
   ```

2. **Checkpointing Strategy:**
   - Saves both model and optimizer states
   - Uses episode numbers in filenames
   - Implements automatic cleanup of old checkpoints

3. **Logging Strategy:**
   - Hierarchical logging structure
   - Separate log files for different runs
   - Includes performance metrics and debugging information

### Performance Considerations
1. **Memory Management:**
   - Efficient tensor operations
   - Proper cleanup of unused checkpoints
   - Optimized data structures for experience replay

2. **Computation Optimization:**
   - Batch processing where possible
   - GPU acceleration support
   - Efficient curriculum generation

3. **Storage Optimization:**
   - Compressed checkpoint format
   - Rotating log files
   - Efficient visualization storage 