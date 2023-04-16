Architecture Decision Record: Decision Making Service
-----------------------------------------------------

### **Status**

Accepted

### **Context**

Our organization is building a decision making service that can be used by other AI services to make decisions based on a given set of criteria. This service will receive inputs from multiple sources and provide an output that can be used as a foundation for other services when they are faced with the making of a decision. The inputs can be fine-tuned models, historical data, user preferences, and other relevant data sources.

### **Decision**

We have decided to build a decision making service that queries multiple fine-tuned models and provides an aggregated output based on the inputs received. The service will use a persistence layer like Weaviate to store and manage the data, and each model will be queried using an LLM (Language Model as a Service) approach.

### **Consequences**

#### **Positive:**

- Our decision making service will be able to make informed decisions based on multiple sources of information and historical data, leading to better outcomes for our customers and clients.
- The LLM approach will make it easy to integrate new models into the service as they become available, allowing us to continuously improve the accuracy and effectiveness of our decision making capabilities.
- Using a persistence layer like Weaviate will ensure that data is managed effectively and securely, and can be easily accessed and analyzed by other services as needed.

#### **Negative:**

- There may be a significant development effort required to build the decision making service and integrate it with other systems and services.
- The accuracy of the decision making service may be affected by the quality and reliability of the data sources used.
- The service may require ongoing maintenance and support to ensure that it continues to operate effectively and efficiently over time.

### **Alternatives**

#### **Option 1:**

Build a decision making service that queries a single fine-tuned model and provides an output based on that model's inputs.

#### **Option 2:**

Build a decision making service that uses a rule-based system to make decisions based on a given set of criteria.

### **Rationale**

Option 1 would be less complex and would require less development effort. However, it would not provide the same level of accuracy or effectiveness as our chosen approach. Option 2 would provide a different type of decision making capability, but would not be as flexible or adaptable as the approach we have chosen.

Our chosen approach provides a more robust and effective solution, allowing us to make informed decisions based on multiple sources of information and historical data. It also allows us to continuously improve our decision making capabilities over time, as new models and data sources become available.

### **Notes**

- The decision making service will be built using a microservices architecture, with each service responsible for a specific function or capability.
- We will use a Kubernetes cluster to manage and orchestrate the various services and components of the system.
- The decision making service will be designed to be scalable and resilient, with built-in redundancy and failover capabilities to ensure that it can continue to operate effectively in the event of hardware or software failures.
- The service will be developed using modern programming languages and frameworks, with a focus on maintainability, testability, and scalability.

Conclusion
----------

Our decision to build a decision making service that queries multiple fine-tuned models and uses a persistence layer like Weaviate represents the best approach for our organization. This approach provides a robust and effective solution for making informed decisions based on multiple sources of information and historical data. It also allows us to continuously improve our decision making capabilities over time, making it a valuable asset for our organization and our customers.
