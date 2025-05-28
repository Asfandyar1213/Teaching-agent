import pytest
import torch
import numpy as np
from agents.teacher import TeachingAgent
from agents.student import StudentAgent
from environments.task_env import TeachingTaskEnv

@pytest.fixture
def env():
    return TeachingTaskEnv(task_complexity=0.5)

@pytest.fixture
def teacher(env):
    return TeachingAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        curriculum_dim=env.state_dim
    )

@pytest.fixture
def student(env):
    return StudentAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim
    )

def test_teacher_initialization(teacher, env):
    """Test teacher agent initialization"""
    assert teacher.state_dim == env.state_dim
    assert teacher.action_dim == env.action_dim
    assert teacher.curriculum_dim == env.state_dim

def test_student_initialization(student, env):
    """Test student agent initialization"""
    assert student.state_dim == env.state_dim
    assert student.action_dim == env.action_dim

def test_teacher_action_selection(teacher):
    """Test teacher's action selection"""
    state = np.random.randn(teacher.state_dim)
    action = teacher.select_action(state)
    assert isinstance(action, int)
    assert 0 <= action < teacher.action_dim

def test_student_action_selection(student):
    """Test student's action selection"""
    state = np.random.randn(student.state_dim)
    action = student.select_action(state)
    assert isinstance(action, int)
    assert 0 <= action < student.action_dim

def test_curriculum_generation(teacher):
    """Test curriculum generation"""
    state = np.random.randn(teacher.state_dim)
    action = teacher.select_action(state)
    curriculum = teacher.generate_curriculum(state, action)
    assert curriculum.shape == (teacher.curriculum_dim,)
    assert isinstance(curriculum, np.ndarray)

def test_environment_interaction(env, teacher, student):
    """Test full interaction cycle"""
    state, _ = env.reset()
    
    # Teacher's turn
    teacher_action = teacher.select_action(state)
    curriculum = teacher.generate_curriculum(state, teacher_action)
    
    # Student's turn
    student_understanding = student.understand_curriculum(state, curriculum)
    student_action = student.select_action(student_understanding)
    
    # Environment step
    next_state, reward, done, _, info = env.step(student_action)
    
    assert isinstance(next_state, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)

def test_agent_updates(teacher, student):
    """Test agent update methods"""
    # Create sample experience
    states = np.random.randn(10, teacher.state_dim)
    actions = np.random.randint(0, teacher.action_dim, 10)
    rewards = np.random.randn(10)
    next_states = np.random.randn(10, teacher.state_dim)
    curricula = np.random.randn(10, teacher.curriculum_dim)
    
    # Test teacher update
    teacher_experience = (states, actions, rewards, next_states)
    teacher_metrics = teacher.update(teacher_experience)
    assert isinstance(teacher_metrics, dict)
    
    # Test student update
    student_experience = (states, actions, rewards, next_states, curricula)
    student_metrics = student.update(student_experience)
    assert isinstance(student_metrics, dict) 