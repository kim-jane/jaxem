import configparser
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import re


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
        
    elif '{' in input and '}' in input:
        parts = re.split(r"(\{.*?\})", input.strip())
        newstring = ''
        for part in parts:
            if part.startswith('{') and part.endswith('}'):
                newstring += str(context[part[1:-1]])
            else:
                newstring += part
        return newstring

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
            
        # dictionary to store parsed values
        self.context = {}

        # set attribute names and values for all input variables
        for section in self.config.sections():
            for name in self.config[section]:
                value = interpret(self.config.get(section, name), self.context)
                setattr(self, name, value)
                self.context[name] = value
                
        #self.write_info('output/'+self.output+'.out')
        self.print_info()
                
    def set(self, name, value):
        setattr(self, name, value)
        self.context[name] = value
           
    '''
    def write_info(self, filename):
        with open(filename, 'w') as file:
            for section in self.config.sections():
                for name in self.config[section]:
                    value = interpret(self.config.get(section, name), self.context)
                    file.write(f'# {name} = {value}\n')

    '''
    def print_info(self):
        for section in self.config.sections():
            for name in self.config[section]:
                value = interpret(self.config.get(section, name), self.context)
                if name == 'Elab':
                    print(f'{name}:')
                    for i in range(len(value)):
                        print(f'\t{i} -> {value[i]} MeV')
                else:
                    print(f'{name} = {value}')

