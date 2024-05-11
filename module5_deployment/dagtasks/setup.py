from setuptools import setup


setup(
    name='dagtasks',
    version='0.1.0',
    description='Functions for airflow dag tasks',
    # package_dir={'': 'dag_tasks/dag_tasks'},
    author='me',
    python_requires='>=3.9',
    install_requires=[
        'numpy>=1.20',
        'pandas>=2.0',
        'requests>=2.1',
        'scikit-learn>=1.2, < 1.4.0',
        'mlflow>=2.9'
    ]
)
