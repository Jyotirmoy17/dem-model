from setuptools import setup, find_packages

setup(
    name='dem',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A glass-box model that distills explanations from a complex model into an interpretable one.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    url='https://github.com/Jyotirmoy17/dem-model',
    install_requires=[
        'scikit-learn',
        'xgboost',
        'numpy'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
