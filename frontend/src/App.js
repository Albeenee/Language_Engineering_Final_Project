import React, {useState, useEffect, Component} from 'react';
import classes from './App.css';
import {createBrowserRouter, RouterProvider} from "react-router-dom";

import HomePage from './pages/Home';
import Trigram from './pages/Trigram';
import Gru from './pages/Gru';
import Lstm from './pages/Lstm';

// import EnterTextTrigram from './components/EnterTextTrigram';

const router = createBrowserRouter([
  {path:'/', element: <HomePage/> },
  {path:'/trigram', element:<Trigram/>},
  {path:'/gru', element:<Gru/>},
  {path:'/lstm', element:<Lstm/>}
])

function App() {
    return (
      <div className={classes.content}>
        <RouterProvider router={router}/>
      </div>
      
    );
  };
 
export default App;

//   return (<>
//   <h1>Autocomplete</h1>
//   <EnterTextTrigram/>
//     </>
//   );
// }

// export default App;
