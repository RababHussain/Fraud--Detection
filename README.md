# Fraud--Detection

As businesses continue to evolve and migrate to the internet and money is transacted electronically in an ever-growing cash- less banking economy, accurate fraud detection remains a key concern for modern banking systems. 
In payment industry, fraud on card occurs when someone steals information from your card to do purchases with- out your permission and the detection of these fraudulent transactions has become a crucial activity for payment processors
In this project, I aim to address these fraudulent activities through sequence classification of credit card numbers using Random Forest and Long Short Term Mmemory (LSTM) aiming to provide an accuracy of 94% in online transactions and 99% in offline transactions.

This system has been developed using Python programming language. A typical fraud detection systems is composed of an automatic tool and a manual process.
The automatic tool is based on fraud detection rules. It analyzes all the new incoming transactions and as- signs a fraudulent score. The manual process is made by fraud investigators.
They focus on transactions with high fraudulent score and provide a binary feedback (fraud or genuine) on all the trans- actions they analyzed. 

#<b>OPERATING ENVIRONMENT:</b>

Windows XP.

#<b>HARDWARE REQUIREMENTS:</b>

•	Processor			-	Pentium –IV

•	Speed				-    	1.1 Ghz

•	RAM				-    	256 MB(min)

•	Hard Disk			-   	20 GB

•	Key Board			-    	Standard Windows Keyboard

•	Mouse				-    	Two or Three Button Mouse

•	Monitor			-    	SVGA

#<b>SOFTWARE REQUIREMENTS:</b>

•	Operating System		-	Windows7/8

•	Programming Language	-	Python 

#<b>CLASS DIAGRAM:</b>
 
![image](https://user-images.githubusercontent.com/96685742/192088591-955990ce-d564-4768-a060-d3632e5b4c21.png)

#<b>USE CASE DIAGRAM:</b>

![image](https://user-images.githubusercontent.com/96685742/192088620-5ba23f48-c97c-49af-aa1f-c9f839e9f419.png)


#<b>SCREENSHOTS:</b>


![image](https://user-images.githubusercontent.com/96685742/192088646-17dd623a-d8e1-42fc-91d1-9a4ec8520972.png)

![image](https://user-images.githubusercontent.com/96685742/192088655-dfe4dff3-fd00-4f43-b0ba-bb5d47a247c8.png)

![image](https://user-images.githubusercontent.com/96685742/192088666-410aaa11-2246-4d69-95b9-40a63d8b3fcb.png)

![image](https://user-images.githubusercontent.com/96685742/192088714-0b74385c-f6d4-4b07-a735-abded0806470.png)

![image](https://user-images.githubusercontent.com/96685742/192088811-8abab612-9a1d-4f1c-9e27-0daa5a3a222f.png)

![image](https://user-images.githubusercontent.com/96685742/192088830-0694811d-28ca-4060-a594-2cccfd6c4bfe.png)

![image](https://user-images.githubusercontent.com/96685742/192088849-e04bc9ee-23af-418b-9bf6-2cd039b0fac4.png)

![image](https://user-images.githubusercontent.com/96685742/192088857-ead3e0db-9e84-4acd-8727-bc2929cdd7b3.png)

#<b>CONCLUSION:</b>

 I have employed long short-term memory networks as a means to aggregate the historic purchase behavior of credit- card holders with the goal to improve fraud detection accuracy on new incoming transactions. 
Compared the results to a baseline classifier that is agnostic to the past. 
This study concludes that offline and online transactions exhibit very different qualities with respect to the sequential character of successive transactions. 
For offline transactions, an LSTM is an adequate model of the latent sequential patterns in that it improves the detection. 
As an alternative to the sequence learner, manual aggregation of the transaction history through additional features improves both the detection on offline and online transactions. 
