# Maximum number of points in each point cloud
MAX_POINTS = 100

FILE_LABEL_MAPPING = {
    '1.1_sit.json': {'Table A': 1, 'Table B': 0, 'Table C': 0},
    '1.1_stand.json': {'Table A': 2, 'Table B': 0, 'Table C': 0},
    '1.2_sit.json': {'Table A': 0, 'Table B': 1, 'Table C': 0},
    '1.2_stand.json': {'Table A': 0, 'Table B': 2, 'Table C': 0},
    '1.3_sit.json': {'Table A': 0, 'Table B': 0, 'Table C': 1},
    '1.3_stand.json': {'Table A': 0, 'Table B': 0, 'Table C': 2},
    '2.1_sit.json': {'Table A': 1, 'Table B': 1, 'Table C': 0},
    '2.1_stand.json': {'Table A': 2, 'Table B': 2, 'Table C': 0},
    '2.2_hybrid.json': {'Table A': 0, 'Table B': 2, 'Table C': 1},
    '3.1_sit.json': {'Table A': 1, 'Table B': 1, 'Table C': 1},
    '3.1_stand.json': {'Table A': 2, 'Table B': 2, 'Table C': 2},
    '3.2_hybrid.json': {'Table A': 1, 'Table B': 1, 'Table C': 2}
}

# Table settings
TABLES = {
    "Table A": {"x_min": -1.2, "x_max": 0.2, "y_min": 0.6, "y_max": 2.0},
    "Table B": {"x_min": -0.2, "x_max": 1.2, "y_min": 2.0, "y_max": 3.2},
    "Table C": {"x_min": -1.5, "x_max": 0.0, "y_min": 3.2, "y_max": 4.5},
}
Z_MIN, Z_MAX = 0.0, 2.0  # Z-axis range is the same for all tables
