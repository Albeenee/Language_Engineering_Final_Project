import React, { Fragment } from 'react';
import {Link} from 'react-router-dom';

// import classes from './Home.module.css';
import Button from '../UI/Button';

const Home = () => {
    console.log('home')
    return (<Fragment>
            <h1>Choose model for word prediction</h1>
            <p></p>
            <Button items={[
                {model:'trigram', link:"/trigram", name:"Trigram"},
                {model:'rnn',link:"/gru", name:"GRU"},
                {model:'lstm',link:"/lstm", name:"LSTM"}]}/>
            </Fragment>
    );
};
export default Home; 