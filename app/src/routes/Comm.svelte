<script>
    // communicate with the backend using websockets
    import { createEventDispatcher } from "svelte";
    const dispatch = createEventDispatcher();
    export let messages = ["m1", "m2", "m3"];
    let new_message;
    const sessionName = "my-session";  // Customize this dynamically
    const signalingServer = new WebSocket("ws://localhost:8080");

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

</script>


<div class="charbar h-8 w-64 bg-gray-500">
    <input bind:value={new_message} type="text" placeholder="Type a message..." />
    <button
        on:click = {() => {
            messages = [...messages, new_message];
            signalingServer.send(JSON.stringify({ sessionName: sessionName, signalData: new_message }));
        }}
    >Send</button>
</div>


<div class="chatbox h-screen w-64 bg-gray-200"
>
    {#each messages as message}
        <p>{message}</p>
    {/each}
</div>