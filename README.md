# Agents That Learn to Teach Other Agents

This project implements an adaptive teaching system where teaching agents learn to generate curricula and guide learning agents through complex tasks. The system uses meta-learning and curriculum learning approaches to create effective teaching strategies.

## Key Features

- Teaching agents that can:
  - Generate adaptive curricula
  - Monitor student performance
  - Adjust teaching strategies in real-time
- Learning agents that:
  - Receive and follow instructions
  - Provide feedback on learning progress
- Dynamic curriculum generation
- Customizable task environments

## Project Structure

```
├── agents/
│   ├── teacher.py      # Teaching agent implementation
│   ├── student.py      # Learning agent implementation
│   └── base.py         # Base agent classes
├── curriculum/
│   ├── generator.py    # Curriculum generation logic
│   └── evaluator.py    # Curriculum evaluation
├── environments/
│   └── task_env.py     # Task environment implementation
├── utils/
│   ├── metrics.py      # Performance metrics
│   └── visualization.py # Learning progress visualization
└── main.py            # Main training script
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main training script:
```bash
python main.py
```

## Implementation Details

The system uses:
- Meta-learning for teaching strategy adaptation
- Curriculum learning for task progression
- Reinforcement learning for both teacher and student agents
- Real-time performance monitoring and strategy adjustment

## License

MIT License 