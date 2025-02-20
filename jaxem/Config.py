'''
import configparser
import jax.numpy as jnp


def interpret(input):
    """
    Turns the input string into a numpy array,
    boolean, float, or integer, if possible.
    """
    symbols = {"+", "-", "/", "*", "pi"}
    
    if 'log' in input:
        input_list = input.strip('log ').split(' ')
        min, max, num = [float(x) for x in input_list]
        return jnp.logspace(min, max, int(num))

    elif 'lin' in input:
        input_list = input.strip('lin ').split(' ')
        min, max, num = [float(x) for x in input_list]
        return jnp.linspace(min, max, int(num))

    elif '[' in input and ']' in input:
        input_list = input.strip('[]').split(',')
        if input_list[0].isdigit():
            return jnp.array(input_list, dtype=jnp.int64)
        else:
            return jnp.array(input_list, dtype=jnp.float64)
        
    elif 'true' in input.lower():
        return True
    
    elif 'false' in input.lower():
        return False

    elif input.isdigit():
        return int(input)
    
    elif any(sym in input for sym in symbols):
        return eval(input.strip(), {"__builtins__": {}}, {"pi": jnp.pi, "jnp": jnp})
        
    elif '.' in input or 'E' in input:
        return float(input)

    else:
        return input


class Config:
    """
    An instance of the Config class will automatically set (key, value)
    pairs listed in the input .ini file as attributes.
    
    Example:
        Suppose 'Example.ini' contains the line:
            MyKey = MyVal
        Then we can construct a Config object like this:
            config = Config('Example.ini')
        And access the (key, value) pair like this:
            config.MyKey = MyVal
            
    The Config class makes it easier to deal with many inputs.
    Instead of passing certain inputs into individual classes,
    the Config object will store all of the inputs, and we can
    "look them up" as needed.
    """

    def __init__(self, file=None):
    
        # create the ConfigParser object
        self.config = configparser.ConfigParser(inline_comment_prefixes='#')
        
        # make the ConfigParser object case-sensitive
        self.config.optionxform = str
        
        # read 'Input.ini' file if no file is provided
        if file == None:
            print('No file provided. Set parameters manually.')
        else:
            print(f'Reading {file}...')
            self.config.read(file)
            
        # set attribute names and values for all input variables
        for section in self.config.sections():
            for name in self.config[section]:
                value = interpret(self.config.get(section, name))
                setattr(self, name, value)

    def set(self, name, value):
        setattr(self, name, value)
            
    def write_info(self):
    
        with open(self.output+'.info', 'w') as file:
            
            for section in self.config.sections():
                for name in self.config[section]:
                    value = interpret(self.config.get(section, name))
                    print(f'{name} = {value}')
                    file.write(f'{name} = {value}')

    def print_info(self):

        for section in self.config.sections():
            for name in self.config[section]:
                value = interpret(self.config.get(section, name))
                print(f'{name} = {value}')

'''

import configparser
import jax.numpy as jnp


def interpret(input, context=None):
    """
    Turns the input string into a numpy array, boolean, float, or integer, if possible.
    Evaluates expressions based on previously defined symbols in `context`.
    """
    symbols = {"+", "-", "/", "*", "pi"}
    context = context or {}

    if 'log' in input:
        input_list = input.strip('log ').split(' ')
        min, max, num = [float(x) for x in input_list]
        return jnp.logspace(min, max, int(num))

    elif 'lin' in input:
        input_list = input.strip('lin ').split(' ')
        min, max, num = [float(x) for x in input_list]
        return jnp.linspace(min, max, int(num))

    elif '[' in input and ']' in input:
        input_list = input.strip('[]').split(',')
        if input_list[0].isdigit():
            return jnp.array(input_list, dtype=jnp.int64)
        else:
            return jnp.array(input_list, dtype=jnp.float64)
        
    elif 'true' in input.lower():
        return True
    
    elif 'false' in input.lower():
        return False

    elif input.isdigit():
        return int(input)
    
    elif any(sym in input for sym in symbols):
        # Evaluate using the stored context
        return eval(input.strip(), {"__builtins__": {}}, {"pi": jnp.pi, "jnp": jnp, **context})

    elif '.' in input or 'E' in input:
        return float(input)

    else:
        return input


class Config:
    """
    Reads parameters from an input .ini file and stores them as attributes.
    Supports evaluating expressions that depend on previously defined variables.
    """

    def __init__(self, file=None):
    
        # create the ConfigParser object
        self.config = configparser.ConfigParser(inline_comment_prefixes='#')
        
        # make the ConfigParser object case-sensitive
        self.config.optionxform = str
        
        # read the input file if provided
        if file is None:
            print('No file provided. Set parameters manually.')
        else:
            print(f'Reading {file}...')
            self.config.read(file)
            
        # Dictionary to store parsed values
        self.context = {}

        # set attribute names and values for all input variables
        for section in self.config.sections():
            for name in self.config[section]:
                value = interpret(self.config.get(section, name), self.context)
                setattr(self, name, value)
                self.context[name] = value  # Store in context for future evaluations

    def set(self, name, value):
        setattr(self, name, value)
        self.context[name] = value  # Update context
            
    def write_info(self):
        with open(self.output + '.info', 'w') as file:
            for section in self.config.sections():
                for name in self.config[section]:
                    value = interpret(self.config.get(section, name), self.context)
                    print(f'{name} = {value}')
                    file.write(f'{name} = {value}')

    def print_info(self):
        for section in self.config.sections():
            for name in self.config[section]:
                value = interpret(self.config.get(section, name), self.context)
                print(f'{name} = {value}')
