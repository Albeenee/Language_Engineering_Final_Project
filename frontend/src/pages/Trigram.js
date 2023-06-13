import React, {Fragment, useEffect,useState} from 'react';
import axios from 'axios';

import Button from '../UI/Button';
import Suggestion from '../UI/Suggestion';
import Form from '../UI/Form';
import KeystrokesCounter from '../UI/KeystrokesCounter';


const Trigram = props => {

    // initialize states
    const [results, setResults] = useState(null);
    const [type, setType] = useState('nextword');
    const [enteredText, setEnteredText] = useState('');
    const [keystrokesCounter, setKeystrokesCounter] = useState(0);
    const [newStrokes, setNewStrokes] = useState(0);
    
    // send input and retrieving output from backend
    const sendDataToBackend = async (text, guesstype) => {
        try {
        console.log('processing...');
        const data = { input: text.toLowerCase(), guesstype:guesstype}; // Your data to send
        const response = await axios.post('/trigram', data);
        console.log(response.data.pred);
        setResults(response.data.pred);

        } catch (error) {
        console.error(error);
        }
      };

    // Input change handling
    const inputChangeHandler = (event) => {
      setEnteredText(event.target.value);
      setNewStrokes(newStrokes+1);
      // if last key is spacebar, set type to 'nextword'
      if (event.target.value.at(-1)===' '){
      setType('nextword');
      sendDataToBackend(event.target.value, 'nextword')
     } else {
      //else, set type to 'endword'
      setType('endword');
      sendDataToBackend(event.target.value, 'endword');
      }
    };

    // Handling when a word is selected
    const selectHandler = (result) => {

      let newtext;
      setKeystrokesCounter(keystrokesCounter+result.length-newStrokes+1);
      setNewStrokes(0);
      
      // prepare to call next word function from backend
      if (type === 'nextword') {
        let ib = '';
        // add space if necessary (managing of async)
        if (enteredText.length > 0 && enteredText.at(-1) !== ' ') {
          ib = ' ';
        }
        newtext = enteredText + ib + result;
      } else {
        const text = enteredText.split(' ');
        text.pop();
        newtext = text.join(' ') + ' ' + result + ' ';
      }
    
      setEnteredText(newtext);
      setType('nextword');
      sendDataToBackend(newtext, 'nextword');
    };

    return (
      <Fragment>
        <h1>Trigram</h1>
        <Form model='trigram' html='input' value={enteredText} inputChangeHandlerFunction={inputChangeHandler}/>
        {results && 
        <Suggestion model='trigram' results = {results} handlefunction={selectHandler}/>}
        {/* <KeystrokesCounter model='trigram' results={keystrokesCounter}/> */}
        <Button items={[{model:'trigram', link:"/", name:"Home"}]}/>
      </Fragment>
    );
}

export default Trigram;