def WrapperExample(Weaviate): 
      
    class Wrapper: 
          
        def __init__(self, y): 
              
            self.wrap = A(y) 
              
        def get_number(self): 
              
 
            return self.wrap.name 
          
    return Wrapper 
  
@decorator
class code: 
    def __init__(self, z): 
        self.name = z 
 
y = code("Wrapper class") 
print(y.get_name()) 