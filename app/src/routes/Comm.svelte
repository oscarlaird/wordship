<script>
    // communicate with the backend using websockets
    import "../app.css";    
    import { createEventDispatcher } from "svelte";
    import { onMount } from "svelte";
    const dispatch = createEventDispatcher();
    export let messages = [];
    let new_message;
    const sessionName = "my-session";  // Customize this dynamically
    // const signalingServer = new WebSocket("ws://localhost:8080");
    // const signalingServer = new WebSocket("wss://142.93.14.74:8080");
    const signalingServer = new WebSocket("wss://domainnamefortesting.com");

    signalingServer.onopen = () => {
        console.log("Connected to signaling server");
        signalingServer.send(JSON.stringify({ sessionName, signalData: "connect_session"}))
    };

    // Listen for incoming signaling messages
    signalingServer.onmessage = async (message) => {
        const { signalData } = JSON.parse(message.data);
        // Handle received signal data (used for WebRTC)
        await handleSignalData(signalData);
    };

    async function handleSignalData(signalData) {
        console.log("Received signal data: ", signalData);
        messages = [...messages, signalData];
        console.log(messages);
        dispatch("receive_message", { message: signalData });
        // Handle signal data
    }
    async function handleKeyPress(event) {
        if (event.key === "Enter" && new_message) {
            messages = [...messages, new_message];
            signalingServer.send(JSON.stringify({ sessionName: sessionName, signalData: new_message }));
            new_message = "";
            event.preventDefault();
        }
    }
    onMount(() => {
        window.addEventListener("keypress", handleKeyPress);
        return () => {
            window.removeEventListener("keypress", handleKeyPress);
        };
    });

</script>
<div class="h-screen w-screen flex flex-col justify-center items-center bg-green-600 p-2">
    <div class="card bg-white p-4 rounded-lg shadow-lg flex flex-col flex-1 w-full h-full overflow-y-scroll gap-2">
        <div class="text-2xl md:text-4xl font-serif flex flex-row justify-center w-full p-2">
            <div class="flex flex-row flex-1 max-w-xl">
                <input bind:value={new_message} type="text" placeholder="Type a word..."
                    class="border border-gray-400 p-2 m-2 rounded-lg flex-1"
                />
                <button class="bg-green-900 text-white p-2 m-2 rounded-lg whitespace-nowrap"
                    on:click = {() => {
                        messages = [...messages, new_message];
                        signalingServer.send(JSON.stringify({ sessionName: sessionName, signalData: new_message }));
                    }}
                >Send</button>
            </div>
        </div>
        <hr>
        <i class="text-sm">Click a message to resend.</i>
        {#each [...messages].reverse() as message}
            <div class="text-lg md:text-2xl break-words"
                on:click={() => {
                    new_message = message;
                    messages = [...messages, new_message];
                    signalingServer.send(JSON.stringify({ sessionName: sessionName, signalData: new_message }));
                }}
            >
                {message}
            </div>
        {/each}
    </div>
</div>
