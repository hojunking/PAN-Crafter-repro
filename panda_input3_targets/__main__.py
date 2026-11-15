"""Run unchanged INPUT3 training with target-qualified Sheet publication."""
from .release import verify
from .reporting import install
from panda_input3.__main__ import main

if __name__ == '__main__':
    verify()
    install()
    main()
