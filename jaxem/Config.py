import configparser
import jax.numpy as jnp


def interpret(input):
    """
    Turns the input string into a numpy array,
    boolean, float, or integer, if possible.
    """
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


