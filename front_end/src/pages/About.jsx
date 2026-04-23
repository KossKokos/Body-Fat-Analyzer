import { Link } from "react-router-dom";

const About = () => {
  return (
    <div className="max-w-5xl mx-auto space-y-8">
      <section className="card">
        <div className="mb-4">
          <span className="inline-flex rounded-full bg-primary-100 px-3 py-1 text-sm font-medium text-primary-700">
            About this project
          </span>
        </div>

        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          Body Fat Percentage Predictor
        </h1>

        <p className="text-lg text-gray-700 leading-8">
          I built this project to show how machine learning work can move beyond
          analysis notebooks and become part of a complete application. The app
          takes health, training, and nutrition inputs, runs them through a
          prediction pipeline, stores the result, and lets the user send
          feedback back into the system.
        </p>
      </section>

      <section className="card">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">
          What I wanted this project to prove
        </h2>

        <div className="space-y-4 text-gray-700 leading-7">
          <p>
            The main goal was not just to train a model and stop there. I wanted
            to build something that shows I understand the full path from data
            work to a usable product.
          </p>

          <p>
            That means working with data in a practical way, thinking carefully
            about the prediction pipeline, and then connecting that pipeline to
            a backend, a database, and a frontend that someone can actually use.
          </p>
        </div>
      </section>

      <section className="card">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">
          Data science and machine learning work
        </h2>

        <div className="space-y-4 text-gray-700 leading-7">
          <p>
            On the machine learning side, this project reflects the kind of work
            that happens before a model is ever exposed through an API. It comes
            from reading the data properly, understanding what each feature is
            doing, cleaning the inputs, thinking about useful transformations,
            and making the data suitable for training and inference.
          </p>

          <p>
            I wanted the prediction flow to feel like a real system rather than
            a single model with one direct output. The final setup combines
            classification and regression, which gave me a way to structure the
            prediction logic in stages instead of treating everything as one flat
            problem.
          </p>

          <p>
            An important part of the project was building the prediction logic so
            it could be reused consistently. I did not want something that only
            worked inside an experiment. I wanted a pipeline that could be loaded
            by the backend, called by the API, and return stable results in the
            same way every time.
          </p>
        </div>

        <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="rounded-xl border border-gray-200 p-4 bg-gray-50">
            <h3 className="font-semibold text-gray-900 mb-2">
              ML work covered here
            </h3>
            <ul className="list-disc pl-5 space-y-2 text-sm text-gray-700">
              <li>working with structured health and fitness data</li>
              <li>cleaning and preparing inputs for training and inference</li>
              <li>feature handling and pipeline consistency</li>
              <li>classification and regression in one prediction flow</li>
              <li>moving from model logic to reusable inference code</li>
            </ul>
          </div>

          <div className="rounded-xl border border-gray-200 p-4 bg-gray-50">
            <h3 className="font-semibold text-gray-900 mb-2">
              What matters to me here
            </h3>
            <ul className="list-disc pl-5 space-y-2 text-sm text-gray-700">
              <li>not treating ML as notebook-only work</li>
              <li>keeping inference logic structured and reusable</li>
              <li>making model behaviour fit into a real application</li>
              <li>thinking about feedback and future improvement loops</li>
            </ul>
          </div>
        </div>
      </section>

      <section className="card">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">
          Turning the model into an actual product
        </h2>

        <div className="space-y-4 text-gray-700 leading-7">
          <p>
            A big part of the value in this project is that the model is not
            isolated. It sits inside a FastAPI backend that handles validation,
            prediction requests, error handling, and persistence. The frontend
            then works with that API as a separate application, which makes the
            whole project closer to a real production setup than a local demo.
          </p>

          <p>
            I also built the feedback flow so the app does not stop at showing a
            prediction. The user can respond to the result, rate it, and provide
            extra context. That part was useful because it forced me to think
            about schema design, state handling, validation, and how prediction
            systems can collect information for future improvement.
          </p>
        </div>
      </section>

      <section className="card">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">
          Database, migrations, and application structure
        </h2>

        <div className="space-y-4 text-gray-700 leading-7">
          <p>
            I used PostgreSQL for persistence and Alembic for migrations, which
            was important because I wanted the data layer to be handled properly
            rather than treated as an afterthought. Prediction history and
            feedback are stored in separate but related tables, so the project
            also reflects an understanding of table structure, relationships, and
            schema changes over time.
          </p>

          <p>
            Working through migrations, model changes, and feedback fields helped
            make the project much more realistic. It stopped being just a
            prediction endpoint and became a small system with state, history,
            and data that needs to be managed cleanly.
          </p>
        </div>
      </section>

      <section className="card">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">
          Security and engineering decisions
        </h2>

        <div className="space-y-4 text-gray-700 leading-7">
          <p>
            I paid attention to the kind of details that matter once a project
            stops being purely experimental. That included request validation,
            handling configuration through environment variables, keeping API
            responses controlled, being careful about what the frontend sends and
            receives, and avoiding unnecessary exposure of internal backend
            details.
          </p>

          <p>
            I also wanted the project structure to stay maintainable. That meant
            separating the frontend and backend properly, keeping the API layer
            clear, using migrations instead of manual schema edits, and making
            the frontend handle loading, errors, results, and feedback in a way
            that feels deliberate rather than patched together.
          </p>
        </div>
      </section>

      <section className="card">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">
          Docker and deployment thinking
        </h2>

        <div className="space-y-4 text-gray-700 leading-7">
          <p>
            I also wanted the project to be easy to run in a controlled
            environment, so Docker is part of the setup. That matters to me
            because it shows I am not only thinking about code that works on my
            own machine, but about repeatability, local setup, and cleaner
            deployment paths.
          </p>

          <p>
            For me, that is part of the same mindset as good ML engineering:
            building something that can actually be run, tested, and maintained,
            not just something that produces a promising result once.
          </p>
        </div>
      </section>

      <section className="card border border-amber-200 bg-amber-50">
        <h2 className="text-2xl font-bold text-amber-900 mb-4">
          Important note
        </h2>

        <div className="space-y-4 text-amber-900 leading-7">
          <p>
            This is a portfolio project. It is here to show how I approach data
            science, machine learning, backend engineering, and full-stack
            delivery in one piece of work.
          </p>

          <p>
            It is not a medical product and it should not be treated as one. Any
            result shown by the app is an estimate included for demonstration
            purposes only.
          </p>
        </div>
      </section>

      <section className="card">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">
          Why this project matters in my portfolio
        </h2>

        <div className="space-y-4 text-gray-700 leading-7">
          <p>
            What matters most to me about this project is that it ties together
            the parts of data work that are often shown separately. It starts
            with data and modelling, but it does not stop there. It carries that
            work into an API, a database, a frontend, and a deployable
            application structure.
          </p>

          <p>
            That is the kind of work I want to keep doing: building machine
            learning systems that are not only technically sound, but also
            usable, maintainable, and grounded in real engineering.
          </p>

          <p>
            You can also read the{" "}
            <Link
              to="/privacy-policy"
              className="font-semibold underline hover:text-primary-600 transition-colors"
            >
              Privacy Policy
            </Link>{" "}
            and{" "}
            <Link
              to="/terms-of-use"
              className="font-semibold underline hover:text-primary-600 transition-colors"
            >
              Terms of Use
            </Link>{" "}
            for the hosted project, or{" "}
            <a
              href="mailto:Kostiantyn.Pereimybida@outlook.com"
              className="font-semibold underline hover:text-primary-600 transition-colors"
            >
              get in touch
            </a>
            .
          </p>
        </div>
      </section>
    </div>
  );
};

export default About;